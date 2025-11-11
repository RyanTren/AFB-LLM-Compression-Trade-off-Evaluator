import os
import json
import re
from codebleu import calc_codebleu

BASE_RESULTS_DIR = "./results"
LANGUAGE = "python"
REFS_PATH = "data/code_references.json"

def clean_generated_code(text):
    """Extract only the first function definition from generated code"""
    # Remove task/solution markers
    text = re.sub(r'#\s*Task:.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'#\s*Solution:.*?\n', '', text, flags=re.IGNORECASE)
    
    # Extract first function
    lines = text.split('\n')
    code_lines = []
    in_function = False
    base_indent = None
    
    for line in lines:
        stripped = line.lstrip()
        
        if stripped.startswith('def '):
            if in_function:
                break  # Stop at second function
            in_function = True
            base_indent = len(line) - len(stripped)
            code_lines.append(line)
        elif in_function:
            if not stripped:  # Empty line
                code_lines.append(line)
            else:
                current_indent = len(line) - len(line.lstrip())
                if current_indent > base_indent:
                    code_lines.append(line)
                else:
                    break  # End of function
    
    result = '\n'.join(code_lines).strip()
    return result if result else text.strip()

def normalize_code(code):
    """Normalize whitespace and indentation"""
    if not code:
        return ""
    lines = code.split('\n')
    normalized = []
    for line in lines:
        if line.strip():
            # Standardize indentation to 4 spaces
            indent_level = (len(line) - len(line.lstrip())) // 4
            normalized.append('    ' * indent_level + line.lstrip())
        else:
            normalized.append('')
    return '\n'.join(normalized).strip()

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
                
                # Clean and normalize
                gen_output = clean_generated_code(gen_output)
                gen_output = normalize_code(gen_output)
                
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
        clean_prompt = prompt.strip().lstrip('#').strip().rstrip('.')
        
        # Try exact match
        if prompt in refs_json:
            refs.append(normalize_code(refs_json[prompt]))
        elif clean_prompt in refs_json:
            refs.append(normalize_code(refs_json[clean_prompt]))
        else:
            # Fuzzy matching
            best_match = None
            best_score = 0
            
            for ref_key in refs_json.keys():
                prompt_words = set(clean_prompt.lower().split())
                ref_words = set(ref_key.lower().replace('#', '').split())
                
                if prompt_words and ref_words:
                    overlap = len(prompt_words & ref_words)
                    score = overlap / max(len(prompt_words), len(ref_words))
                    
                    if score > best_score and score > 0.3:
                        best_score = score
                        best_match = ref_key
            
            if best_match:
                refs.append(normalize_code(refs_json[best_match]))
            else:
                refs.append("")
    
    return refs

def evaluate_folder(folder_path, ref_path):
    prompts, generated = load_results(folder_path)
    if not generated:
        print(f"⚠️ No results found in {folder_path}")
        return None
    
    refs = load_references(ref_path, prompts)
    print(f"🔍 Evaluating {len(generated)} samples in {os.path.basename(folder_path)}...")

    # Debug first sample
    if generated and refs:
        print(f"\n📝 Sample 0 (cleaned):")
        print(f"Reference:\n{refs[0][:150]}")
        print(f"\nGenerated:\n{generated[0][:150]}\n")

    scores = []
    for i, (ref, hyp) in enumerate(zip(refs, generated)):
        if not ref or not hyp:
            scores.append(0.0)
            print(f"  Sample {i}: SKIPPED (empty)")
            continue
            
        try:
            # Try real CodeBLEU
            result = calc_codebleu([ref], [hyp], lang=LANGUAGE)
            score = result["codebleu"]
            scores.append(score)
            print(f"  Sample {i}: CodeBLEU = {score:.4f}")
        except Exception as e:
            # Fallback to token similarity
            ref_tokens = set(ref.split())
            hyp_tokens = set(hyp.split())
            
            if ref_tokens and hyp_tokens:
                intersection = len(ref_tokens & hyp_tokens)
                union = len(ref_tokens | hyp_tokens)
                score = intersection / union if union > 0 else 0.0
            else:
                score = 0.0
            
            scores.append(score)
            print(f"  Sample {i}: Jaccard = {score:.4f} (fallback: {str(e)[:50]})")

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score

def main():
    summary = {}
    for subdir in sorted(os.listdir(BASE_RESULTS_DIR)):
        folder_path = os.path.join(BASE_RESULTS_DIR, subdir)
        if not os.path.isdir(folder_path):
            continue
        avg = evaluate_folder(folder_path, REFS_PATH)
        if avg is not None:
            summary[subdir] = avg
            print(f"✅ {subdir}: Average = {avg:.4f}")
        print("=" * 60)

    # Save summary
    summary_path = os.path.join(BASE_RESULTS_DIR, "codebleu_summary_all.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📊 Saved summary to: {summary_path}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()