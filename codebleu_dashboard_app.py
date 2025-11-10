import json, os, subprocess, tempfile, sys, re
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

# ---------- Config ----------
API_URL_DEFAULT = "http://localhost:8510/generate"
CODEBLEU_DIR = str(Path("~/mnt/cst/gpu4/sperry46/p10-t1llmcomp/CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU").expanduser())
CALC_SCRIPT = str(Path(CODEBLEU_DIR) / "calc_code_bleu.py")
SPLIT_TOKEN = "<END_OF_SNIPPET>"
# ----------------------------

st.set_page_config(page_title="CodeBLEU Dashboard", layout="wide")
st.title("CodeBLEU Evaluation (GPTQ 4‑bit)")

with st.sidebar:
    st.header("Server & Data")
    api_url = st.text_input("Serving API URL", API_URL_DEFAULT)
    jsonl_path = st.text_input("Eval JSONL (prompt/reference)", str(Path("~/eval/eval.jsonl").expanduser()))
    lang = st.selectbox("Language", ["python","java","javascript","go","cpp","csharp"], index=0)

    st.header("Generation Params")
    max_new_tokens = st.number_input("max_new_tokens", min_value=1, max_value=4096, value=128)
    temperature = st.number_input("temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    top_p = st.number_input("top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

    st.header("Batching")
    limit = st.number_input("Evaluate first N examples (0 = all)", min_value=0, max_value=100000, value=50)

    st.header("Scoring")
    default_has_parser = Path(CODEBLEU_DIR).joinpath("parser", "my-languages.so").exists()
    include_syntax = st.checkbox("Include syntax & data-flow (requires parser)", value=default_has_parser)

st.write(
    "**Instructions**: The app calls your `/generate` API for each prompt, writes references & hypotheses "
    "to temp files, then invokes the CodeBLEU `calc_code_bleu.py` script and visualizes results."
)

def load_examples(p: str):
    items = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def call_api(prompt: str) -> str:
    payload = {
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    r = requests.post(api_url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"]

run_eval = st.button("Run Evaluation")

if run_eval:
    # --- preflight ---
    if not Path(jsonl_path).exists():
        st.error(f"Eval file not found: {jsonl_path}")
        st.stop()
    if not Path(CALC_SCRIPT).exists():
        st.error(f"CodeBLEU script not found at {CALC_SCRIPT}. "
                 f"Set CODEBLEU_DIR or place calc_code_bleu.py there.")
        st.stop()

    # Probe CodeBLEU CLI once (from the CodeBLEU folder to resolve local imports)
    help_txt = ""
    try:
        help_txt = subprocess.run(
            [sys.executable, CALC_SCRIPT, "-h"],
            capture_output=True, text=True, cwd=CODEBLEU_DIR
        ).stdout.lower()
    except Exception:
        pass

    # Figure out --ref vs --refs and whether --split_token / --gamma/--theta are supported
    ref_flag = "--refs" if "--refs" in help_txt or "--ref " not in help_txt else "--ref"
    supports_split = "--split_token" in help_txt
    supports_weights = ("--gamma" in help_txt and "--theta" in help_txt)

    # Load and optionally limit examples
    exs = load_examples(jsonl_path)
    if limit and limit > 0:
        exs = exs[:limit]
    st.write(f"Loaded {len(exs)} examples.")

    # --- generate hypotheses ---
    refs, hyps, rows = [], [], []
    total = max(len(exs), 1)
    pb = st.progress(0)
    for i, ex in enumerate(exs, 1):
        prompt = ex.get("prompt", "")
        ref = ex.get("reference", "")
        try:
            hyp = call_api(prompt)
        except Exception as e:
            hyp = f"[ERROR: {e}]"
        refs.append(ref.rstrip("\n"))
        hyps.append(hyp.rstrip("\n"))
        rows.append({"idx": i, "prompt": prompt, "reference": ref, "hypothesis": hyp})
        pb.progress(int(i * 100 / total))

    df = pd.DataFrame(rows)
    st.subheader("Samples")
    st.dataframe(df, use_container_width=True)

    # --- write refs/hyps and run CodeBLEU ---
    with tempfile.TemporaryDirectory() as td:
        ref_p = Path(td) / "ref.txt"
        hyp_p = Path(td) / "hyp.txt"
        with open(ref_p, "w", encoding="utf-8") as rf, open(hyp_p, "w", encoding="utf-8") as hf:
            for r, h in zip(refs, hyps):
                rf.write(r + "\n" + SPLIT_TOKEN + "\n")
                hf.write(h + "\n" + SPLIT_TOKEN + "\n")

        cmd = [sys.executable, CALC_SCRIPT, ref_flag, str(ref_p), "--hyp", str(hyp_p), "--lang", lang]
        if supports_split:
            cmd += ["--split_token", SPLIT_TOKEN]
        if not include_syntax and supports_weights:
            cmd += ["--gamma", "0", "--theta", "0"]

        st.write("Running CodeBLEU…")
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=CODEBLEU_DIR)
        if out.returncode != 0:
            st.error("CodeBLEU failed:\n\n" + (out.stderr or out.stdout))
            st.stop()
        st.code(out.stdout + ("\n" + out.stderr if out.stderr else ""), language="text")

        # --- parse scores robustly ---
        bleu = weighted = syntax = dataflow = codebleu = None
        for line in out.stdout.splitlines():
            t = line.strip()
            tl = t.lower()
            if re.search(r'(^|\s)codebleu( score)?\s*[:=]\s*([0-9.]+)', tl):
                codebleu = float(re.findall(r'([0-9.]+)\s*$', t)[-1])
            elif re.search(r'(^|\s)weighted[\s-]*ngram match( score)?\s*[:=]\s*([0-9.]+)', tl):
                weighted = float(re.findall(r'([0-9.]+)\s*$', t)[-1])
            elif re.search(r'(^|\s)(syntactic|syntax) match( score)?\s*[:=]\s*([0-9.]+)', tl):
                syntax = float(re.findall(r'([0-9.]+)\s*$', t)[-1])
            elif re.search(r'(^|\s)dataflow match( score)?\s*[:=]\s*([0-9.]+)', tl):
                dataflow = float(re.findall(r'([0-9.]+)\s*$', t)[-1])
            elif re.search(r'(^|\s)bleu(\s*score)?\s*[:=]\s*([0-9.]+)', tl):
                bleu = float(re.findall(r'([0-9.]+)\s*$', t)[-1])

        st.subheader("Scores")
        score_rows = []
        if bleu is not None:     score_rows.append(("BLEU", bleu))
        if weighted is not None: score_rows.append(("Weighted n‑gram", weighted))
        if syntax is not None:   score_rows.append(("Syntax", syntax))
        if dataflow is not None: score_rows.append(("Data‑flow", dataflow))
        if codebleu is not None: score_rows.append(("CodeBLEU", codebleu))

        if score_rows:
            sdf = pd.DataFrame(score_rows, columns=["Metric", "Score"])
            st.bar_chart(sdf.set_index("Metric"))
            st.dataframe(sdf, hide_index=True)
        else:
            st.warning("Could not parse CodeBLEU output. See raw output above.")

        st.download_button("Download predictions (CSV)", df.to_csv(index=False), "predictions.csv", "text/csv")
