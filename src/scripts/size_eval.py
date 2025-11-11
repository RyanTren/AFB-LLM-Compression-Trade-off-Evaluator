from pathlib import Path
import torch
from tabulate import tabulate  # pip install tabulate

# Path containing all your LoRA folders
LORA_ROOT = Path("./")  # adjust if needed

# Collect results for summary
summary = []

# Loop over all LoRA folders
for lora_folder in LORA_ROOT.glob("lora_out_*"):
    if not lora_folder.is_dir():
        continue

    largest_size = 0
    largest_ckpt = None
    largest_params = 0

    for ckpt in lora_folder.glob("checkpoint-*"):
        pytorch_file = ckpt / "pytorch_model.bin"
        if pytorch_file.exists():
            size_mb = pytorch_file.stat().st_size / 1024**2
            adapter_weights = torch.load(pytorch_file, map_location="cpu")
            total_params = sum(v.numel() for v in adapter_weights.values())

            if size_mb > largest_size:
                largest_size = size_mb
                largest_ckpt = ckpt.name
                largest_params = total_params

    if largest_ckpt:
        summary.append({
            "LoRA Folder": lora_folder.name,
            "Largest Checkpoint": largest_ckpt,
            "LoRA Params": f"{largest_params:,}",
            "Size (MB)": f"{largest_size:.2f}",
        })

# Sort summary by size descending
summary_sorted = sorted(summary, key=lambda x: float(x["Size (MB)"]), reverse=True)

# Print table
print(tabulate(summary_sorted, headers="keys", tablefmt="github"))
