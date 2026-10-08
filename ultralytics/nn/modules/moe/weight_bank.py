"""Per-image weight-bank routing for 1x1 convolutions.

A weight bank holds ``E`` kernels for one 1x1 convolution. One router per network reads a pooled feature and
returns a gate vector per image; every banked convolution then runs a single convolution whose kernel is the gated
sum of its bank. Because a 1x1 convolution is linear in its kernel, mixing kernels is exact:
``sum_i g_i * conv(x, W_i) == conv(x, sum_i g_i * W_i)``. The executed multiply-accumulates therefore equal those
of the dense layer, while the layer holds ``E`` times the parameters and reads ``k / E`` of them per image.

The module is applied to an already built model with :func:`convert_to_weight_bank`, driven by the ``weight_bank``
key of the model YAML. Batch normalization and the activation stay in the surrounding ``Conv`` block, after the
mix, so the block's state-dict keys are unchanged apart from the kernel itself.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ("BankConv2d", "GateState", "WeightBankRouter", "bank_modules", "convert_to_weight_bank", "fit_fixed_router")

LOWERINGS = (None, "conv", "matmul", "conv_local", "matmul_local")


class GateState:
    """Mutable state shared by one router and every banked convolution it drives.

    The object is a plain Python object on purpose: holding it as an attribute does not register it as a submodule,
    and ``copy.deepcopy`` / ``pickle`` keep it shared between the modules of one model copy.

    Attributes:
        num_experts (int): Number of kernels per bank.
        top_k (int): Number of kernels mixed per image.
        hard (bool): Mix the selected kernels with equal weights (``1 / top_k``) instead of a softmax over their
            logits. A hard top-1 selects one kernel; a hard top-2 averages a pair. Both deploy as static branches.
        members (list[BankConv2d]): Banked convolutions in forward order.
        gates (torch.Tensor | None): Dense gate matrix ``[B, E]`` of the current forward pass.
        logits (torch.Tensor | None): Detached router logits ``[B, E]`` of the current forward pass.
        lowering (str | None): ``None`` for the training/eval path, otherwise the static batch-1 deployment graph.
            ``"conv"`` / ``"matmul"`` gather once over the concatenated bank and slice the mixed vector per layer;
            ``"conv_local"`` / ``"matmul_local"`` gather from each layer's own bank. The suffix-free part names the
            tensor-kernel operation that consumes the mixed kernel.
        mixed (torch.Tensor | None): Flat mixed kernel vector of the current deployment forward pass.
        selected (tuple | None): Selected expert indices and their weights for the per-layer lowerings.

    Examples:
        >>> state = GateState(num_experts=4, top_k=2)
        >>> state.num_experts, state.top_k, state.lowering
        (4, 2, None)
    """

    def __init__(self, num_experts: int, top_k: int, hard: bool = False):
        """Initialize the shared state."""
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got top_k={top_k}, num_experts={num_experts}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.hard = hard
        self.members = []
        self.gates = None
        self.logits = None
        self.lowering = None
        self.mixed = None
        self.selected = None
        self._flat = None

    def __getstate__(self):
        """Drop per-forward tensors so checkpoints do not carry them."""
        state = self.__dict__.copy()
        state.update(gates=None, logits=None, mixed=None, selected=None, _flat=None)
        return state

    def offsets(self) -> list[int]:
        """Return the start offset of every member in the concatenated bank."""
        out, total = [], 0
        for m in self.members:
            out.append(total)
            total += m.out_channels * m.in_channels
        return out

    def flat_bank(self) -> torch.Tensor:
        """Return the concatenated bank ``[E, total]`` used by the deployment graph (cached)."""
        if self._flat is None:
            self._flat = torch.cat([torch.stack(tuple(m.experts)).flatten(1) for m in self.members], 1).detach()
        return self._flat


class WeightBankRouter(nn.Module):
    """Router for weight-bank routing: global average pool, two-layer MLP, top-k softmax gate.

    The router always runs in float32. In the training/eval path it returns a dense ``[B, E]`` gate matrix that is a
    softmax over the ``top_k`` largest logits and zero elsewhere, so each row sums to one.

    The pooled feature is standardised per channel with running statistics before the MLP (a batch normalization
    without learnable parameters). Pooled backbone features differ between images by a small fraction of their mean,
    so without it the logits are nearly the same for every image at initialisation. The normalization has no weight
    or bias, hence nothing for weight decay to shrink, and folds into the first linear layer at export.

    Args:
        c1 (int): Channels of the routed feature map.
        num_experts (int): Number of kernels per bank.
        top_k (int): Number of kernels mixed per image.
        hidden (int): Width of the hidden layer; 0 gives a linear router (one layer), the form of a nearest-centroid
            classifier, which :func:`fit_fixed_router` can set from k-means centroids.

    Attributes:
        fixed (bool): When True the router is a fixed partition: its parameters do not train and the running
            statistics are used in training as well (see :meth:`fix`).

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> router = WeightBankRouter(c1=16, num_experts=4, top_k=2)
        >>> gates = router(torch.randn(3, 16, 8, 8))
        >>> gates.shape, int((gates > 0).sum(1).max()), bool(torch.allclose(gates.sum(1), torch.ones(3)))
        (torch.Size([3, 4]), 2, True)
    """

    def __init__(self, c1: int, num_experts: int = 4, top_k: int = 2, hidden: int = 64):
        """Initialize the router."""
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got top_k={top_k}, num_experts={num_experts}")
        self.c1, self.num_experts, self.top_k = c1, num_experts, top_k
        self.fixed = False
        self.norm = nn.BatchNorm1d(c1, affine=False)
        self.fc1 = nn.Linear(c1, hidden) if hidden else nn.Identity()
        self.act = nn.SiLU() if hidden else nn.Identity()
        self.fc2 = nn.Linear(hidden or c1, num_experts)

    def fix(self) -> WeightBankRouter:
        """Freeze the router as a fixed partition: no parameter updates, running statistics in every mode."""
        self.fixed = True
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return float32 router logits ``[B, E]`` for a 4-D NCHW feature map."""
        if x.ndim != 4:
            raise ValueError(f"WeightBankRouter expects a 4-D NCHW tensor, got {x.ndim} dimensions")
        if x.shape[1] != self.c1:
            raise ValueError(f"WeightBankRouter expects {self.c1} channels, got {x.shape[1]}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            pooled = x.float().mean((2, 3))
            n = self.norm
            if self.training and not self.fixed and pooled.shape[0] > 1:  # one image cannot provide batch statistics
                pooled = F.batch_norm(pooled, n.running_mean, n.running_var, None, None, True, n.momentum, n.eps)
            else:  # plain arithmetic, so the exported graph carries no normalization operator
                pooled = (pooled - n.running_mean) * torch.rsqrt(n.running_var + n.eps)
            return self.fc2(self.act(self.fc1(pooled)))

    @staticmethod
    def gates_from_logits(logits: torch.Tensor, top_k: int, hard: bool = False) -> torch.Tensor:
        """Return the dense gate matrix: softmax (or equal weights if ``hard``) over the ``top_k`` largest logits."""
        if top_k >= logits.shape[1] and not hard:
            return logits.softmax(1)
        vals, idx = logits.topk(top_k, dim=1)
        weights = torch.full_like(vals, 1.0 / top_k) if hard else vals.softmax(1)
        return torch.zeros_like(logits).scatter(1, idx, weights)

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Return the dense gate matrix ``[B, E]``."""
        return self.gates_from_logits(self.logits(x), self.top_k, hard)


class BankConv2d(nn.Module):
    """A 1x1 convolution whose kernel is a per-image gated sum of ``E`` kernels.

    The module replaces the ``nn.Conv2d`` inside a ``Conv`` block. Kernels are stored as ``E`` separate parameters
    of the dense shape ``[c2, c1, 1, 1]`` so optimizers treat each one like the dense kernel. The first banked
    convolution in forward order owns the router and reads its own input as the router feature.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        state (GateState): State shared with the router and the other banked convolutions.
        router (WeightBankRouter, optional): Router owned by this module.
        bias (bool): Whether the convolution has a bias (set when batch normalization is folded in).

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> state = GateState(num_experts=4, top_k=2)
        >>> conv = BankConv2d(8, 6, state, router=WeightBankRouter(8, 4, 2))
        >>> x = torch.randn(2, 8, 5, 5)
        >>> y = conv(x)
        >>> y.shape
        torch.Size([2, 6, 5, 5])
        >>> ref = sum(g.view(-1, 1, 1, 1) * F.conv2d(x, w) for g, w in zip(state.gates.T, conv.experts))
        >>> bool(torch.allclose(y, ref, atol=1e-5))
        True
    """

    def __init__(self, c1: int, c2: int, state: GateState, router: WeightBankRouter | None = None, bias: bool = False):
        """Initialize the bank with independently initialised kernels."""
        super().__init__()
        self.in_channels, self.out_channels = c1, c2
        self.kernel_size, self.stride, self.padding, self.dilation, self.groups = (1, 1), (1, 1), (0, 0), (1, 1), 1
        self.experts = nn.ParameterList(
            nn.Parameter(nn.Conv2d(c1, c2, 1, bias=False).weight.detach().clone()) for _ in range(state.num_experts)
        )
        self.bias = nn.Parameter(torch.zeros(c2)) if bias else None
        self.router = router
        self.state = state
        self.index = len(state.members)
        state.members.append(self)

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, state: GateState, router: WeightBankRouter | None = None, eps: float = 0.01):
        """Build a bank from a dense 1x1 convolution.

        Every kernel starts as the dense kernel. With ``eps > 0`` a perturbation is added to each; the
        perturbations sum to zero over the bank and each has about ``eps`` times the norm of the dense kernel, so
        uniform gates reproduce the dense layer exactly while the router receives a non-zero gradient from the first
        step. With ``eps == 0`` the kernels are exact copies. With ``eps=None`` the kernels keep their independent
        random initialisation (training from scratch).

        Args:
            conv (nn.Conv2d): Dense 1x1 convolution with stride 1, groups 1 and no padding.
            state (GateState): Shared state.
            router (WeightBankRouter, optional): Router owned by the new module.
            eps (float | None): Relative norm of the perturbation, 0 for exact copies, None for independent kernels.

        Returns:
            (BankConv2d): The banked convolution.

        Examples:
            >>> _ = torch.manual_seed(0)
            >>> dense = nn.Conv2d(8, 6, 1, bias=False)
            >>> bank = BankConv2d.from_conv(dense, GateState(4, 2), eps=0.01)
            >>> mean = torch.stack(tuple(bank.experts)).mean(0)
            >>> bool(torch.allclose(mean, dense.weight, atol=1e-6))
            True
        """
        if not is_bankable(conv):
            raise ValueError("BankConv2d.from_conv expects a 1x1 convolution with stride 1, groups 1 and no padding")
        bank = cls(conv.in_channels, conv.out_channels, state, router, bias=conv.bias is not None)
        bank.to(device=conv.weight.device, dtype=conv.weight.dtype)
        with torch.no_grad():
            if eps is not None:
                noise = torch.randn(state.num_experts, *conv.weight.shape, device=conv.weight.device)
                noise -= noise.mean(0, keepdim=True)
                noise *= eps * conv.weight.norm() / noise.flatten(1).norm(dim=1).mean().clamp_min(1e-12)
                for kernel, delta in zip(bank.experts, noise):
                    kernel.copy_(conv.weight + delta.to(conv.weight.dtype))
            if conv.bias is not None:
                bank.bias.copy_(conv.bias)
        return bank

    def fuse_batchnorm(self, bn: nn.BatchNorm2d) -> None:
        """Fold a following batch normalization into every kernel and the shared bias (called by ``model.fuse``)."""
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shift = bn.bias - bn.running_mean * scale
        with torch.no_grad():
            for kernel in self.experts:
                kernel.mul_(scale.view(-1, 1, 1, 1).to(kernel.dtype))
            bias = shift if self.bias is None else self.bias * scale + shift
        self.bias = nn.Parameter(bias.detach().clone().to(self.experts[0].dtype))
        self.state._flat = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the gated convolution."""
        state = self.state
        if state.lowering is not None:
            return self._forward_deploy(x)
        if self.router is not None:
            logits = self.router.logits(x)
            state.logits = logits.detach()
            state.gates = self.router.gates_from_logits(logits, state.top_k, state.hard)
        b, _, h, w = x.shape
        gates = state.gates
        if gates is None or gates.shape[0] != b:
            raise RuntimeError("BankConv2d ran without gates for this batch; the router must run first")
        with torch.autocast(device_type=x.device.type, enabled=False):
            bank = torch.stack(tuple(self.experts)).flatten(1).float()
            kernel = (gates.float() @ bank).view(b, self.out_channels, self.in_channels)
        y = torch.bmm(kernel.to(x.dtype), x.flatten(2)).view(b, self.out_channels, h, w)
        return y if self.bias is None else y + self.bias.view(1, -1, 1, 1).to(y.dtype)

    def _forward_deploy(self, x: torch.Tensor) -> torch.Tensor:
        """Static batch-1 graph: one gather over the concatenated bank, then one tensor-kernel operation."""
        state = self.state
        if x.shape[0] != 1:
            raise RuntimeError("the weight-bank deployment graph is defined for batch size 1")
        local = state.lowering.endswith("_local")
        if self.router is not None:
            logits = self.router.logits(x)
            state.logits = logits.detach()
            if state.top_k >= state.num_experts and not state.hard:  # soft mixing reads the whole bank
                idx, weights = None, logits.softmax(1)
            elif state.top_k == 1:
                idx, weights = logits.argmax(1), None
            else:
                vals, idx = logits.topk(state.top_k, dim=1)
                idx = idx[0]
                weights = torch.full_like(vals, 1.0 / state.top_k) if state.hard else vals.softmax(1)
            state.selected = (idx, weights)
            if not local:
                flat = state.flat_bank().to(x.device)
                rows = flat if idx is None else flat[idx]
                state.mixed = rows if weights is None else weights.to(flat.dtype) @ rows
        if local:
            idx, weights = state.selected
            rows = torch.stack(tuple(self.experts)).flatten(1)
            rows = rows if idx is None else rows[idx]
            kernel = (rows if weights is None else weights.to(rows.dtype) @ rows).flatten().to(x.dtype)
        else:
            start = state.offsets()[self.index]
            kernel = state.mixed[0, start : start + self.out_channels * self.in_channels].to(x.dtype)
        if state.lowering.startswith("conv"):
            return F.conv2d(x, kernel.view(self.out_channels, self.in_channels, 1, 1), self.bias)
        _, _, h, w = x.shape
        y = torch.matmul(kernel.view(self.out_channels, self.in_channels), x.flatten(2))
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)
        return y.view(1, self.out_channels, h, w)

    def extra_repr(self) -> str:
        """Return a short description for ``print(model)``."""
        owner = ", router" if self.router is not None else ""
        return f"{self.in_channels}, {self.out_channels}, experts={len(self.experts)}, top_k={self.state.top_k}{owner}"


def is_bankable(conv: nn.Module) -> bool:
    """Return True for a dense 1x1 convolution that can be replaced by a weight bank.

    Examples:
        >>> is_bankable(nn.Conv2d(4, 4, 1)), is_bankable(nn.Conv2d(4, 4, 3, padding=1)), is_bankable(nn.Identity())
        (True, False, False)
    """
    return (
        isinstance(conv, nn.Conv2d)
        and conv.kernel_size == (1, 1)
        and conv.stride == (1, 1)
        and conv.padding == (0, 0)
        and conv.groups == 1
    )


def bank_modules(model: nn.Module) -> list[BankConv2d]:
    """Return the banked convolutions of a model in forward order.

    Examples:
        >>> bank_modules(nn.Conv2d(4, 4, 1))
        []
    """
    banks = [m for m in model.modules() if isinstance(m, BankConv2d)]
    return sorted(banks, key=lambda m: m.index)


def convert_to_weight_bank(
    layers: nn.Sequential,
    first: int,
    last: int,
    experts: int = 4,
    top_k: int = 2,
    hidden: int = 64,
    eps: float = 0.01,
    hard: bool = False,
    router_fixed: bool = False,
) -> GateState:
    """Replace every dense 1x1 convolution in ``layers[first:last + 1]`` by a weight bank, in place.

    Only convolutions wrapped in a ``Conv`` block (convolution, batch normalization, activation) are converted. The
    first converted convolution owns the router, so the router reads the input of layer ``first``.

    Args:
        layers (nn.Sequential): Layer list of a built model (``model.model``).
        first (int): Index of the first routed layer.
        last (int): Index of the last routed layer (inclusive).
        experts (int): Number of kernels per bank.
        top_k (int): Number of kernels mixed per image.
        hidden (int): Hidden width of the router (0: linear router).
        eps (float | None): Perturbation norm (see :meth:`BankConv2d.from_conv`): 0 exact copies, None independent.
        hard (bool): Equal-weight mixing of the selected kernels (see :class:`GateState`).
        router_fixed (bool): Freeze the router as a fixed partition (see :meth:`WeightBankRouter.fix`).

    Returns:
        (GateState): The state shared by the converted layers.

    Examples:
        >>> from ultralytics.nn.modules.conv import Conv
        >>> _ = torch.manual_seed(0)
        >>> net = nn.Sequential(Conv(3, 8, 3), Conv(8, 8, 1), Conv(8, 4, 1))
        >>> x = torch.randn(2, 3, 16, 16)
        >>> dense = net.eval()(x)
        >>> state = convert_to_weight_bank(net, first=1, last=2, experts=4, top_k=4, eps=0.01)
        >>> len(state.members), type(net[1].conv).__name__
        (2, 'BankConv2d')
        >>> for p in net[1].conv.router.fc2.parameters():  # zero logits give uniform gates: the dense network
        ...     _ = nn.init.zeros_(p)
        >>> bool(torch.allclose(net(x), dense, atol=1e-5))
        True
    """
    from ultralytics.nn.modules.conv import Conv

    state = GateState(experts, top_k, hard)
    for layer in list(layers)[first : last + 1]:
        for block in layer.modules():
            if type(block) is Conv and is_bankable(block.conv):
                router = None
                if not state.members:
                    router = WeightBankRouter(block.conv.in_channels, experts, top_k, hidden)
                    router.to(block.conv.weight.device)
                    if router_fixed:
                        router.fix()
                block.conv = BankConv2d.from_conv(block.conv, state, router, eps)
    if not state.members:
        raise ValueError(f"no 1x1 Conv block found in layers {first}..{last}")
    return state


def fit_fixed_router(
    router: WeightBankRouter, pooled: torch.Tensor, iters: int = 50, balance: bool = True, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set a linear router to a fixed, balanced nearest-centroid partition of pooled features, and freeze it.

    The router's standardisation statistics are taken from ``pooled``; k-means (``num_experts`` centroids) runs on
    the standardised features; the linear layer is set to the nearest-centroid classifier
    (``logit_i = c_i . z - |c_i|^2 / 2``); with ``balance`` the biases are then adjusted so that every centroid
    receives an equal share of ``pooled``. Finally :meth:`WeightBankRouter.fix` is applied.

    Args:
        router (WeightBankRouter): A linear router (``hidden=0``).
        pooled (torch.Tensor): Pooled router-input features ``[N, c1]`` of representative training images.
        iters (int): k-means iterations.
        balance (bool): Equalise the shares by bias adjustment.
        seed (int): Seed for the centroid initialisation.

    Returns:
        (tuple[torch.Tensor, torch.Tensor]): Shares per expert before and after balancing, measured on ``pooled``.

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> router = WeightBankRouter(8, num_experts=4, top_k=1, hidden=0)
        >>> feats = torch.randn(400, 8) + torch.randint(0, 4, (400, 1)).float() * 3  # four clusters
        >>> before, after = fit_fixed_router(router, feats)
        >>> bool(after.max() - after.min() < 0.08), router.fixed, router.fc2.weight.requires_grad
        (True, True, False)
    """
    if not isinstance(router.fc1, nn.Identity):
        raise TypeError("fit_fixed_router needs a linear router (hidden=0)")
    with torch.no_grad():
        x = pooled.float()
        router.norm.running_mean.copy_(x.mean(0))
        router.norm.running_var.copy_(x.var(0, unbiased=False))
        z = (x - router.norm.running_mean) * torch.rsqrt(router.norm.running_var + router.norm.eps)
        k = router.num_experts
        g = torch.Generator(device=z.device).manual_seed(seed)
        centroids = z[torch.randperm(len(z), generator=g, device=z.device)[:k]].clone()
        for _ in range(iters):
            assign = torch.cdist(z, centroids).argmin(1)
            for i in range(k):
                if (assign == i).any():
                    centroids[i] = z[assign == i].mean(0)
        router.fc2.weight.copy_(centroids)
        router.fc2.bias.copy_(-0.5 * centroids.pow(2).sum(1))
        logits = z @ centroids.T + router.fc2.bias
        before = torch.bincount(logits.argmax(1), minlength=k).float() / len(z)
        after = before.clone()
        if balance:
            step = 0.1 * logits.std()
            for _ in range(500):
                after = torch.bincount(logits.argmax(1), minlength=k).float() / len(z)
                if (after - 1.0 / k).abs().max() < 0.005:
                    break
                router.fc2.bias.add_(step * (1.0 / k - after))
                logits = z @ centroids.T + router.fc2.bias
    router.fix()
    return before, after
