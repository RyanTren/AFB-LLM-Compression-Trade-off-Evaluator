import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ---------- Config ----------
API_URL_DEFAULT = "http://localhost:8000/generate"  # server is on the same VM
CODEBLEU_DIR = str(Path("~/home/p10-t1llmcomp/CodeBLEU").expanduser())
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
    max_new_tokens = st.number_input("max_new_tokens", 1, 2048, 128)
    temperature = st.number_input("temperature", 0.0, 2.0, 0.0, step=0.1)   # deterministic default
    top_p = st.number_input("top_p", 0.0, 1.0, 1.0, step=0.05)              # deterministic default

    st.header("Batching")
    limit = st.number_input("Evaluate first N examples (0 = all)", 0, 100000, 50)

st.write("**Instructions**: The app will call your `/generate` API for each prompt, write references & "
         "hypotheses to temp files, then invoke the CodeXGLUE `calc_code_bleu.py` script and visualize results.")

def load_examples(p):
    items = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            items.append(ex)
    return items

def call_api(prompt):
    payload = {"prompt": prompt, "max_new_tokens": int(max_new_tokens),
               "temperature": float(temperature), "top_p": float(top_p)}
    r = requests.post(api_url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"]

run_eval = st.button("Run Evaluation")

if run_eval:
    if not Path(jsonl_path).exists():
        st.error(f"Eval file not found: {jsonl_path}")
        st.stop()
    if not Path(CALC_SCRIPT).exists():
        st.error(f"CodeBLEU script not found at {CALC_SCRIPT}. Clone CodeXGLUE to {CODEBLEU_DIR}.")
        st.stop()

    exs = load_examples(jsonl_path)
    if limit and limit > 0:
        exs = exs[:limit]
    st.write(f"Loaded {len(exs)} examples.")

    # Generate
    refs, hyps, rows = [], [], []
    pb = st.progress(0)
    for i, ex in enumerate(exs, 1):
        prompt = ex["prompt"]
        ref = ex["reference"]
        try:
            hyp = call_api(prompt)
        except Exception as e:
            hyp = f"[ERROR: {e}]"
        refs.append(ref.rstrip("\n"))
        hyps.append(hyp.rstrip("\n"))
        rows.append({"idx": i, "prompt": prompt, "reference": ref, "hypothesis": hyp})
        pb.progress(i / len(exs))

    df = pd.DataFrame(rows)
    st.subheader("Samples")
    st.dataframe(df, use_container_width=True)

    # Write temp files with a separator between multi‑line snippets
    with tempfile.TemporaryDirectory() as td:
        ref_p = Path(td) / "ref.txt"
        hyp_p = Path(td) / "hyp.txt"
        with open(ref_p, "w", encoding="utf-8") as rf, open(hyp_p, "w", encoding="utf-8") as hf:
            for r, h in zip(refs, hyps):
                rf.write(r + "\n" + SPLIT_TOKEN + "\n")
                hf.write(h + "\n" + SPLIT_TOKEN + "\n")

        # Run CodeBLEU (subprocess)
        cmd = ["python", CALC_SCRIPT, "--refs", str(ref_p), "--hyp", str(hyp_p),
               "--lang", lang, "--split_token", SPLIT_TOKEN]
        st.write("Running CodeBLEU…")
        out = subprocess.run(cmd, capture_output=True, text=True)
        st.code(out.stdout + ("\n" + out.stderr if out.stderr else ""), language="text")

        # Try to parse the four component scores if present
        # (The script prints them; this parser is resilient to formatting.)
        bleu, weighted, syntax, dataflow, codebleu = None, None, None, None, None
        for line in out.stdout.splitlines():
            t = line.strip().lower()
            if "bleu" in t and "weighted" not in t and bleu is None:
                bleu = float(t.split()[-1])
            if "weighted ngram match" in t:
                weighted = float(t.split()[-1])
            if "syntactic match" in t or "syntax match" in t:
                syntax = float(t.split()[-1])
            if "dataflow match" in t:
                dataflow = float(t.split()[-1])
            if "codebleu" in t:
                try:
                    codebleu = float(t.split()[-1])
                except: 
                    pass

        # Visualize
        st.subheader("Scores")
        score_rows = []
        if bleu is not None:     score_rows.append(("BLEU", bleu))
        if weighted is not None: score_rows.append(("Weighted n‑gram", weighted))
        if syntax is not None:   score_rows.append(("Syntax", syntax))
        if dataflow is not None: score_rows.append(("Data‑flow", dataflow))
        if codebleu is not None: score_rows.append(("CodeBLEU", codebleu))

        if score_rows:
            sdf = pd.DataFrame(score_rows, columns=["Metric","Score"])
            st.bar_chart(sdf.set_index("Metric"))
            st.dataframe(sdf, hide_index=True)
        else:
            st.warning("Could not parse CodeBLEU output. See raw output above.")

        # Allow download of CSV with hypotheses
        st.download_button("Download predictions (CSV)", df.to_csv(index=False), "predictions.csv", "text/csv")
