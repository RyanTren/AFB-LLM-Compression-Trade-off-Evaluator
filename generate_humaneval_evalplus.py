import requests
from evalplus.data import get_human_eval_plus, write_jsonl  # HumanEval+
from tqdm import tqdm  # pip install tqdm (optional but nice)

# Your two serving containers
BASE_URL  = "http://localhost:8511/generate"  # un-quantized model
QUANT_URL = "http://localhost:8000/generate"  # quantized model
# If your route is /generate instead of root, change to "...:8511/generate" etc.

def call_model(prompt: str, url: str) -> str:
    """Call your HTTP API and return the raw completion text."""
    payload = {
        "prompt": prompt,
        "max_new_tokens": 128,  # you can bump this if your server allows it
        # For evaluation we want deterministic outputs:
        "temperature": 0.2,
        "top_p": 1.0,
    }
    r = requests.post(url, json=payload, timeout=12000000)
    r.raise_for_status()
    data = r.json()
    return data["response"]  # matches your schema


def gen_samples(url: str, out_path: str):
    """
    Generate HumanEval+ completions with the model served at `url`
    and save them to `out_path` in EvalPlus format.
    """
    problems = get_human_eval_plus()  # dict[task_id] -> problem dict :contentReference[oaicite:1]{index=1}
    samples = []

    for task_id, problem in tqdm(problems.items(), desc=f"Generating for {out_path}"):
        prompt = problem["prompt"]      # HumanEval prompt: def + docstring
        completion = call_model(prompt, url)
        samples.append({
            "task_id": task_id,
            # EvalPlus expects either `solution` OR `completion`.
            # Here we use `completion` = function body / continuation. :contentReference[oaicite:2]{index=2}
            "completion": completion,
        })

    write_jsonl(out_path, samples)
    print(f"Wrote {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    # Base (un-quantized) model
    gen_samples(BASE_URL,  "humaneval_base.jsonl")

    # Quantized model
    gen_samples(QUANT_URL, "humaneval_quant.jsonl")
