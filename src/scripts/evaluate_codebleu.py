import os
import json
from codebleu import calc_codebleu

BASE_RESULTS_DIR = "./results"
LANGUAGE = "python"
REFS_PATH = "data/code_references.json"

def load_results(dir_path):
    """Load all result_*.json files in a directory"""
    texts = []
    prompts = []
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(dir_path, fname)
        try:
            with open(path, "r") as f:
                content = f.read()
            parts = content.split("Generated Solution:")
            if len(parts) == 2:
                prompt = parts[0].replace("Prompt:", "").strip()
                gen_output = parts[1].strip()
                prompts.append(prompt)
                texts.append(gen_output)
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
    return prompts, texts


def load_references(ref_path, prompts):
    """Load reference solutions for the given prompts"""
    with open(ref_path, "r") as f:
        refs_json = json.load(f)
    refs = [refs_json.get(p, "") for p in prompts]
    return refs


def evaluate_folder(folder_path, ref_path):
    prompts, generated = load_results(folder_path)
    if not generated:
        print(f"⚠️ No results found in {folder_path}")
        return None
    refs = load_references(ref_path, prompts)
    print(f"🔍 Evaluating {len(generated)} samples in {os.path.basename(folder_path)}...")

    scores = []
    for ref, hyp in zip(refs, generated):
        try:
            # Try with language parameter (newer API)
            result = calc_codebleu([ref], [hyp], lang=LANGUAGE)
            scores.append(result["codebleu"])
        except TypeError:
            try:
                # Fallback: try without lang parameter and handle manually
                # This uses a simpler BLEU calculation if tree-sitter fails
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                smooth = SmoothingFunction().method1
                
                ref_tokens = ref.split()
                hyp_tokens = hyp.split()
                
                # Calculate BLEU as fallback
                if ref_tokens and hyp_tokens:
                    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
                else:
                    score = 0.0
                    
                scores.append(score)
                print(f"⚠️ Using BLEU fallback (tree-sitter issue)")
            except Exception as e:
                print(f"⚠️ Error calculating score: {e}")
                scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score


def main():
    summary = {}
    for subdir in os.listdir(BASE_RESULTS_DIR):
        folder_path = os.path.join(BASE_RESULTS_DIR, subdir)
        if not os.path.isdir(folder_path):
            continue
        avg = evaluate_folder(folder_path, REFS_PATH)
        if avg is not None:
            summary[subdir] = avg
            print(f"✅ {subdir}: CodeBLEU = {avg:.4f}")
        print("------------------------------------------------")

    # Save summary
    summary_path = os.path.join(BASE_RESULTS_DIR, "codebleu_summary_all.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n📊 Saved global summary to:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()