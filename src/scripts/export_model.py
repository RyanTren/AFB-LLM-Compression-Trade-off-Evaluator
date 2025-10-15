from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, "lora_out")
model = model.merge_and_unload()
model.save_pretrained("merged_gpt2_lora")
