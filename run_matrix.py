import matplotlib

matplotlib.use('Agg')
import os
import time
import torch
import pandas as pd
from ultralytics import YOLO

experiments = {
    "brain-tumor": {
        "data": "ultralytics/cfg/datasets/brain-tumor.yaml",
        "cfg": "examples/lora_examples/yolo_master_brain_tumor_lora.yaml"
    },
    "visdrone": {
        "data": "ultralytics/cfg/datasets/VisDrone.yaml",
        "cfg": "examples/lora_examples/yolo_master_visdrone_lora.yaml"
    }
}

ranks = [4, 8, 16]
epochs = 20
final_results = []
report_path = "matrix_report.md"  # 最终生成的安全报告文件


print("启动升级版防闪退矩阵测试...")

for name, exp in experiments.items():
    for r in ranks:
        print(f"\n================ [运行中] {name} | Rank: {r} ================")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        model = YOLO("yolov8n.pt")

        results = model.train(
            data=exp["data"],
            cfg=exp["cfg"],
            epochs=epochs,
            batch=16,
            imgsz=640,
            lora_r=r,
            lora_alpha=2 * r,
            name=f"matrix_{name}_r{r}",
            workers=0,
        )

        end_time = time.time()
        duration_min = (end_time - start_time) / 60
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

        metrics = results.results_dict
        map50 = metrics.get("metrics/mAP50(B)", 0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0)

        res_entry = {
            "场景": name, "Rank": r, "可训练参数量": f"{trainable_params:,}",
            "训练时间(分钟)": f"{duration_min:.2f}", "峰值显存(GB)": f"{peak_vram_gb:.2f}",
            "mAP50": f"{map50:.4f}", "mAP50-95": f"{map50_95:.4f}"
        }
        final_results.append(res_entry)

        df_temp = pd.DataFrame(final_results)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# YOLO-Master LoRA 垂类场景矩阵测试报告\n\n")
            f.write(df_temp.to_markdown(index=False))
        print(f"数据已实时备份至 {report_path} (mAP50-95: {map50_95:.4f})")

print(f"\n全部测试完成！请直接在 PyCharm 左侧双击打开 `{report_path}` 查看结果！")


