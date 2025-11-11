from pathlib import Path
import torch
from tabulate import tabulate  # pip install tabulate

# Root paths to search for LoRA folders
SEARCH_PATHS = [Path("./"), Path("./models")]

# Collect results for summary
summary = []

for root in SEARCH_PATHS:
    print("Searching in:", root.resolve())
    for lora_folder in root.rglob("lora_out_*"):
        if not lora_folder.is_dir():
            continue

        largest_size = 0
        largest_ckpt = None
        largest_params = 0

        for ckpt in lora_folder.glob("checkpoint-*"):
            # Recursively search for any .bin file inside checkpoint folder
            bin_files = list(ckpt.rglob("*.bin"))
            if not bin_files:
                continue  # no binary weights found

            # Choose the first .bin file (usually the adapter weights)
            pytorch_file = bin_files[0]

            size_mb = pytorch_file.stat().st_size / 1024**2
            adapter_weights = torch.load(pytorch_file, map_location="cpu")
            total_params = sum(v.numel() for v in adapter_weights.values())

            if size_mb > largest_size:
                largest_size = size_mb
                largest_ckpt = ckpt.name
                largest_params = total_params

        if largest_ckpt:
            summary.append({
                "LoRA Folder": str(lora_folder.relative_to(Path("./"))),
                "Largest Checkpoint": largest_ckpt,
                "LoRA Params": f"{largest_params:,}",
                "Size (MB)": f"{largest_size:.2f}",
            })

# Sort summary by size descending
summary_sorted = sorted(summary, key=lambda x: float(x["Size (MB)"]), reverse=True)

# Print table
if summary_sorted:
    print(tabulate(summary_sorted, headers="keys", tablefmt="github"))
else:
    print("No LoRA checkpoints found in the specified paths.")
