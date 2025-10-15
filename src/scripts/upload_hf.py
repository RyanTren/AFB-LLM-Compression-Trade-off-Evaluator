# pip install huggingface_hub
# huggingface-cli login  # paste your HF token

from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="lora_out",
    repo_id="your-username/gpt2-lora-adapter",
    repo_type="model"
)

# after running this script you can load the model from anywhere with:
# from peft import PeftModel
# model = PeftModel.from_pretrained("gpt2", "your-username/gpt2-lora-adapter")
