# Research from Ryan for open-source models that align with use-cases provided from IPs
### [DeepSeek-Coder-V2](https://arxiv.org/html/2406.11931v1?utm_source=chatgpt.com)	
- Purpose-built for coding; strong on reasoning and code tasks; broad language coverage.
- Good choice when you want maximum quality per token locally.
- Technical report + repo claim SOTA-level results on coding/math across many benchmarks; designed as an open model for coding use cases. 
- Multiple sizes (e.g., 16B/236B MoE variants reported). Runs offline; open weights on GitHub. 


### [Qwen2.5-Coder](https://arxiv.org/pdf/2409.12186)
- Dedicated coder family with strong results on code generation and code repair/debugging (e.g., MdEval); good small→large scale options.
- Tech report and blog show SOTA among open models of the same size; public notes on multi-language code repair (MdEval). 
- 7B→32B “Instruct” options; open weights, easy to run locally and quantize. 


### [StarCoder2](https://arxiv.org/pdf/2402.19173)
- Trained largely on permissively-licensed code; competitive on standard coding benchmarks; widely adopted tooling.
- Good baseline for translation/IDE workflows.
- Paper + NVIDIA write-up: StarCoder2-15B outperforms peers at its size and rivals larger models. 
- 3B/7B/15B; strong cost-quality balance; open weights for fully offline use. 

## Metrics
Look into execution-based metrics as primary. Keep CodeBLEU as a secondary signal for translation quality.

- Primary: Compile-success rate, unit-test pass@1 (and pass@k if you sample), runtime/latency, peak memory.
- Secondary: CodeBLEU & BLEU for translation quality; exact-match rate; for APR: plausible vs. correct patch counts.

## Code-to-Code Translation (Refactoring Across Languages)
- **Primary metric:** Execution-based accuracy  
  - Compile success rate  
  - Unit tests pass rate  
- **Secondary metrics:**  
  - **CodeBLEU** (captures syntax, structure, and semantics better than plain BLEU)  
  - BLEU for baseline comparison  
- **Recommended dataset/benchmark:** **CodeXGLUE Java↔C#**  
  - Report BLEU + CodeBLEU + compile accuracy  

## Debugging / Automated Program Repair (APR)
- **Primary metric:** Test-based correctness  
  - Correct-patch rate (fully fixes the bug)  
  - Plausible-patch rate (passes tests but may not be semantically correct)  
  - Regression rate (introduces new bugs)  
- **Recommended datasets:**  
  - **Defects4J** (Java, real-world bugs)  
  - **QuixBugs** (Java/Python, algorithmic bugs)  
  - **MdEval** (multilingual debugging: APR, code review, bug identification)  
- **Note:** Use held-out project splits (esp. in Defects4J) to avoid overfitting and ensure robustness.

## Why Not Only CodeBLEU?
- CodeBLEU is useful for *similarity* and correlates with human judgment.  
- But it may over-credit code that **looks right but doesn’t compile or pass tests**.  
- **Conclusion:** Use **execution/compilation metrics as primary**, with **CodeBLEU as a secondary quality signal**.

## Results to show
Save full prompts, seeds, and patches; log toolchains and hardware. Provide a small dashboard/table showing quality vs. cost so Robins AFB sees the trade-offs directly. (The slides already emphasize cost/size trade-offs and dynamic model selection.)