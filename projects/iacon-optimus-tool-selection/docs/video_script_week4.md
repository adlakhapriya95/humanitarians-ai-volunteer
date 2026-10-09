# Humanitarians AI — Weekly Video Script (Week 4)
**Project:** Generalized Fine-Tuning Pipeline for Jev/Laya Decision Models (RLCD Strategy 1)  
**Speaker Voice:** Bella (`af_bella` — Kokoro TTS)  
**Visual Style:** Canonical Brutalist Cream (`#F4EFEA`), Espresso Ink (`#2A241D`), Terracotta Accent (`#D97757`)  
**Resolution:** 1080p (1920x1080) | 30 FPS  
**Estimated Duration:** ~70 seconds  

---

### Scene 1: Title & Milestone Overview (~10s)
- **Visuals:**
  - Background: Canonical warm cream (`#F4EFEA`) with hairline border.
  - Badge: `@HumanitariansAI` chip in terracotta (`#D97757`).
  - Serif Headline: **"Laya & Jev Fine-Tuning Pipeline"**
  - Subtitle: *"RLCD Strategy 1: Full 60,000 Dataset Scaling & Calibration"*
  - Key Tags: `ModernBERT 421M` • `Brier Scoring Rule` • `Cloud GPU Benchmark`
- **Narration (Bella):**
  > "Good morning! This is Priya with the Week 4 progress update for Humanitarians AI. This week, we completed the full end-to-end validation of our generalized fine-tuning pipeline for the Laya and Jev model family, scaling from local testing all the way to a full 60,000 dataset run on a cloud GPU."

---

### Scene 2: Pipeline Hardening & Universal Adapter (~12s)
- **Visuals:**
  - Header: **"Pipeline Hardening & Universal Ingestion"**
  - 3 Elevated Cards (`#FAF7F2`):
    - Card 1: *Bug Fixes* — Gated HF dataset fallback, cache paths, dict context serialization.
    - Card 2: *Universal Adapter* — Auto-detects tool calling (xLAM), emergency clinical triage, and generic NLP schemas.
    - Card 3: *80/20 Partitioning* — Strict held-out test split for unbiased evaluation.
- **Narration (Bella):**
  > "We audited the training runtime and resolved four critical blockers—including automated fallbacks for gated datasets and type-safe dictionary serialization. We also built an auto-dataset adapter so the pipeline can ingest function calling, medical triage, or custom tabular data without code changes, alongside a strict 80-20 train and test split."

---

### Scene 3: Mathematical Tests & Local Scaling (~13s)
- **Visuals:**
  - Header: **"Local Mathematical & Scaling Validation"**
  - Left Card: *49 Unit Tests Passing* — Brier reward math, strictly proper scoring rule proofs, $\sigma$-annealing.
  - Right Card: *Progressive Scaling on CPU*:
    - 30 examples $\to$ 83.3% Accuracy (1.4 min)
    - 150 examples $\to$ 83.3% Accuracy (4.1 min)
    - 600 examples $\to$ **95.0% Accuracy** (17.1 min, ECE: 2.57%)
- **Narration (Bella):**
  > "All underlying mathematics were verified through our 49-test suite. Testing locally across 30, 150, and 600 examples showed accuracy climbing to 95% while cutting calibration error down to 2.5%. While this proved strong scaling, local CPU throughput showed us that training the full 60,000 dataset required cloud GPU acceleration."

---

### Scene 4: The 60,000 Cloud GPU Breakthrough (~16s)
- **Visuals:**
  - Header: **"Full 60,000 Dataset Run on NVIDIA T4 GPU"**
  - Large Metric Highlight Banner:
    - **99.10% Test Accuracy** (+$14.25\%$ gain on 12,000 unseen queries)
    - **0.83% Expected Calibration Error** (down from 9.98% baseline)
    - **99.11% F1-Score** • **66.14 Minutes Runtime**
  - Badge: `Live on Google Colab`
- **Narration (Bella):**
  > "We moved the pipeline to an NVIDIA T4 GPU on Google Colab, training all 60,000 examples in just 66 minutes. Evaluated on 12,000 strictly held-out test queries, accuracy surged from 84.8% to 99.1%, while calibration error plummeted to 0.83%—producing a model that is both highly accurate and completely honest about its confidence."

---

### Scene 5: The Ablation Proof: Why RLCD Matters (~12s)
- **Visuals:**
  - Header: **"Ablation Study: Base vs. SFT vs. RLCD"**
  - 3-Column Comparative Card:
    - *Raw Base:* 73.3% Acc | 27.3% ECE (Uncalibrated)
    - *Standard SFT:* 93.3% Acc | 5.8% ECE (Greedy labels)
    - *RLCD Strategy 1:* **99.1% Acc** | **0.83% ECE** (Calibrated probability ceiling)
- **Narration (Bella):**
  > "To verify the exact role of reinforcement learning, we ran a side-by-side ablation comparing raw weights, standard fine-tuning, and RLCD. The results showed that while standard fine-tuning improves raw accuracy, RLCD is the critical mechanism that eliminates overconfident mistakes."

---

### Scene 6: Outro & Artifact Deliverables (~8s)
- **Visuals:**
  - Header: **"PR Submitted & Artifacts Live on GitHub"**
  - Callout Links:
    - `AmeyaPadwad/fine_tuner` (Branch: `feat/strategy-1-pipeline-and-tests`)
    - `docs/PIPELINE_VALIDATION_REPORT.md`
    - `Google Colab GPU Notebook`
  - Footer: `@HumanitariansAI • Iacon Optimus Program`
- **Narration (Bella):**
  > "The Pull Request is submitted to Ameya's repository, the interactive Colab notebook is live, and our complete validation report is on GitHub. Thanks for watching, and see you next week!"
