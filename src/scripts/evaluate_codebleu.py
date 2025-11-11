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
    """Load reference solutions for the given prompts with fuzzy matching"""
    with open(ref_path, "r") as f:
        refs_json = json.load(f)
    
    refs = []
    for prompt in prompts:
        # Clean the prompt (remove # and extra whitespace)
        clean_prompt = prompt.strip().lstrip('#').strip().rstrip('.')
        
        # Try exact match first
        if prompt in refs_json:
            refs.append(refs_json[prompt])
        elif clean_prompt in refs_json:
            refs.append(refs_json[clean_prompt])
        else:
            # Try fuzzy matching - look for key similarity
            best_match = None
            best_score = 0
            
            for ref_key in refs_json.keys():
                # Simple word overlap matching
                prompt_words = set(clean_prompt.lower().split())
                ref_words = set(ref_key.lower().split())
                
                if prompt_words and ref_words:
                    overlap = len(prompt_words & ref_words)
                    score = overlap / max(len(prompt_words), len(ref_words))
                    
                    if score > best_score and score > 0.3:  # At least 30% word overlap
                        best_score = score
                        best_match = ref_key
            
            if best_match:
                refs.append(refs_json[best_match])
                print(f"  📎 Matched '{clean_prompt[:50]}...' -> '{best_match[:50]}...'")
            else:
                refs.append("")  # No match found
                print(f"  ⚠️ No reference found for: '{clean_prompt[:60]}'")
    
    return refs


def evaluate_folder(folder_path, ref_path):
    prompts, generated = load_results(folder_path)
    if not generated:
        print(f"⚠️ No results found in {folder_path}")
        return None
    refs = load_references(ref_path, prompts)
    print(f"🔍 Evaluating {len(generated)} samples in {os.path.basename(folder_path)}...")

    # Debug: Show first sample
    if generated and refs:
        print(f"\n📝 DEBUG - First sample:")
        print(f"Prompt: {prompts[0][:100]}...")
        print(f"Reference length: {len(refs[0])} chars")
        print(f"Generated length: {len(generated[0])} chars")
        print(f"Reference preview: {refs[0][:100]}...")
        print(f"Generated preview: {generated[0][:100]}...\n")

    scores = []
    for i, (ref, hyp) in enumerate(zip(refs, generated)):
        try:
            # Try with language parameter (newer API)
            result = calc_codebleu([ref], [hyp], lang=LANGUAGE)
            score = result["codebleu"]
            scores.append(score)
            print(f"  Sample {i}: CodeBLEU = {score:.4f}")
        except (TypeError, AttributeError) as e:
            # Simple token-based similarity as fallback
            ref_tokens = set(ref.split())
            hyp_tokens = set(hyp.split())
            
            if ref_tokens and hyp_tokens:
                # Jaccard similarity
                intersection = len(ref_tokens & hyp_tokens)
                union = len(ref_tokens | hyp_tokens)
                score = intersection / union if union > 0 else 0.0
            else:
                score = 0.0
            
            scores.append(score)
            print(f"  Sample {i}: Jaccard = {score:.4f} (fallback)")

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