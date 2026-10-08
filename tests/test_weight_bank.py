"""Weight-bank routing: exact kernel mixing, conversion, fusion, deployment graph and optimizer grouping."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.moe.weight_bank import (
    BankConv2d,
    GateState,
    WeightBankRouter,
    bank_modules,
    convert_to_weight_bank,
)
from ultralytics.nn.tasks import DetectionModel


def _net():
    torch.manual_seed(0)
    net = nn.Sequential(Conv(3, 16, 3), Conv(16, 16, 1), Conv(16, 8, 1, act=False), Conv(8, 8, 3))
    for m in net.modules():  # non-trivial batch-norm statistics so that fusion is a real test
        if isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0.5, 1.5)
            nn.init.uniform_(m.bias, -0.5, 0.5)
            nn.init.uniform_(m.running_mean, -0.5, 0.5)
            nn.init.uniform_(m.running_var, 0.5, 1.5)
    return net.eval()


def _zero_logits(state):
    for p in state.members[0].router.fc2.parameters():
        nn.init.zeros_(p)


def test_mixing_is_exact():
    """One convolution with the gated kernel equals the gated sum of the per-kernel convolutions."""
    torch.manual_seed(0)
    state = GateState(num_experts=4, top_k=2)
    conv = BankConv2d(8, 6, state, router=WeightBankRouter(8, 4, 2))
    x = torch.randn(5, 8, 7, 7)
    y = conv(x)
    ref = sum(g.view(-1, 1, 1, 1) * F.conv2d(x, w) for g, w in zip(state.gates.T, conv.experts))
    assert torch.allclose(y, ref, atol=1e-5)
    assert int((state.gates > 0).sum(1).max()) == 2
    assert torch.allclose(state.gates.sum(1), torch.ones(5))


def test_router_rejects_wrong_input():
    router = WeightBankRouter(8, 4, 2)
    with pytest.raises(ValueError, match="4-D NCHW"):
        router(torch.randn(2, 8))
    with pytest.raises(ValueError, match="channels"):
        router(torch.randn(2, 4, 3, 3))


def test_router_single_image_in_training_mode():
    """A single image uses running statistics instead of failing on batch statistics."""
    router = WeightBankRouter(8, 4, 2).train()
    assert router(torch.randn(1, 8, 3, 3)).shape == (1, 4)


def test_conversion_reproduces_dense_network_under_uniform_gates():
    net = _net()
    x = torch.randn(3, 3, 16, 16)
    dense = net(x)
    state = convert_to_weight_bank(net, first=1, last=2, experts=4, top_k=4, eps=0.01)
    assert [type(net[i].conv).__name__ for i in range(4)] == ["Conv2d", "BankConv2d", "BankConv2d", "Conv2d"]
    assert [m.router is not None for m in bank_modules(net)] == [True, False]
    _zero_logits(state)
    assert torch.allclose(net(x), dense, atol=1e-5)
    for bank in state.members:  # perturbations have about 1% of the dense norm and differ between kernels
        stack = torch.stack(tuple(bank.experts))
        rel = (stack - stack.mean(0)).flatten(1).norm(dim=1) / stack.mean(0).norm()
        assert torch.allclose(rel, torch.full((4,), 0.01), rtol=0.2)


def test_fusion_and_deployment_graph_match_eval_path():
    net = _net()
    convert_to_weight_bank(net, first=1, last=2, experts=4, top_k=2, eps=0.05)
    x = torch.randn(4, 3, 16, 16)
    ref = net(x)
    fused = copy.deepcopy(net)
    for m in fused.modules():  # what BaseModel.fuse does for a Conv block that wraps an adapter convolution
        if isinstance(m, Conv) and isinstance(m.conv, BankConv2d):
            m.conv.fuse_batchnorm(m.bn)
            del m.bn
            m.forward = m.forward_fuse
    assert torch.allclose(fused(x), ref, atol=1e-5)
    state = bank_modules(fused)[0].state
    for top_k in (1, 2, 4):
        state.top_k = top_k
        for lowering in ("conv", "matmul", "conv_local", "matmul_local"):
            state.lowering = None
            eval_out = fused(x[:1])
            state.lowering = lowering
            assert torch.allclose(fused(x[:1]), eval_out, atol=1e-5), (top_k, lowering)
    with pytest.raises(RuntimeError, match="batch size 1"):
        fused(x)


def test_state_is_shared_after_deepcopy_and_dropped_tensors_on_pickle():
    net = _net()
    state = convert_to_weight_bank(net, first=1, last=2)
    net(torch.randn(2, 3, 16, 16))
    clone = copy.deepcopy(net)
    states = {id(m.state) for m in bank_modules(clone)}
    assert len(states) == 1 and id(state) not in states
    assert state.__getstate__()["gates"] is None


def test_yaml_builds_weight_bank_model():
    model = DetectionModel("yolo26n-wb.yaml", ch=3, nc=80, verbose=False)
    dense = DetectionModel("yolo26n.yaml", ch=3, nc=80, verbose=False)
    banks = bank_modules(model)
    routed = sum(b.out_channels * b.in_channels for b in banks)
    router = sum(p.numel() for p in banks[0].router.parameters())
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert len(banks) == 39
    assert count(model) == count(dense) + 3 * routed + router
    out = model.eval()(torch.zeros(2, 3, 64, 64))
    assert out[0].shape[0] == 2


def _trainer():
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(moe_router_lr_scale=0.5, lora_lr_mult=1.0, warmup_bias_lr=0.1)
    trainer.data = {"nc": 80}
    trainer.adapter_controller = SimpleNamespace(active=False)
    return trainer


@pytest.mark.parametrize("router_decay", [None, 0.0])
def test_bank_kernels_are_ordinary_weights_and_router_decay_is_optional(router_decay):
    """Expert kernels decay like any weight; the router group keeps the global decay unless `router_decay` is given."""

    class Fixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 1)
            self.bank = BankConv2d(4, 4, GateState(4, 2), router=WeightBankRouter(4, 4, 2))

    model = Fixture()
    kwargs = {} if router_decay is None else {"router_decay": router_decay}
    opt = _trainer().build_optimizer(model, name="SGD", lr=0.01, momentum=0.9, decay=0.002, iterations=100, **kwargs)
    names = {id(p): n for n, p in model.named_parameters()}
    groups = {g["param_group"]: g for g in opt.param_groups}
    weight_group = {names[id(p)] for p in groups["weight"]["params"]}
    assert {f"bank.experts.{i}" for i in range(4)} <= weight_group
    assert groups["weight"]["weight_decay"] == pytest.approx(0.002)
    router_group = {names[id(p)] for p in groups["router"]["params"]}
    assert router_group == {
        "bank.router.fc1.weight",
        "bank.router.fc1.bias",
        "bank.router.fc2.weight",
        "bank.router.fc2.bias",
    }
    assert groups["router"]["weight_decay"] == pytest.approx(0.002 if router_decay is None else router_decay)
    assert groups["router"]["lr"] == pytest.approx(0.005)


def test_hard_gates_and_fixed_linear_router():
    """Hard gates weight the selected kernels equally; a fixed linear router never trains and uses running stats."""
    from ultralytics.nn.modules.moe.weight_bank import fit_fixed_router

    torch.manual_seed(0)
    state = GateState(num_experts=4, top_k=2, hard=True)
    router = WeightBankRouter(8, 4, 2, hidden=0)
    conv = BankConv2d(8, 6, state, router=router)
    conv(torch.randn(5, 8, 7, 7))
    assert torch.allclose(state.gates[state.gates > 0], torch.full((10,), 0.5))
    feats = torch.randn(300, 8) + torch.randint(0, 4, (300, 1)).float() * 3
    _, after = fit_fixed_router(router, feats)
    assert after.max() - after.min() < 0.1 and router.fixed
    assert all(not p.requires_grad for p in router.parameters())
    router.train()
    x = torch.randn(6, 8, 3, 3)
    a = router.logits(x)
    router.eval()
    assert torch.allclose(a, router.logits(x))  # running statistics in both modes
    state.top_k = 1
    net = _net()
    dense = net(torch.zeros(1, 3, 16, 16))
    st = convert_to_weight_bank(
        net, first=1, last=2, experts=4, top_k=2, hidden=0, eps=0.0, hard=True, router_fixed=True
    )
    assert st.hard and st.members[0].router.fixed and isinstance(st.members[0].router.fc1, nn.Identity)
    assert torch.allclose(net(torch.zeros(1, 3, 16, 16)), dense, atol=1e-5)  # eps 0: exact copies of the dense kernel
    for bank in st.members:
        assert all(torch.equal(k, bank.experts[0]) for k in bank.experts)
    x = torch.randn(2, 3, 16, 16)
    ref = net(x)
    st.lowering = "conv"
    assert torch.allclose(net(x[:1]), ref[:1], atol=1e-5)  # the deployment graph mixes the pair with equal weights
