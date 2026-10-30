# Oct 26 – Oct 30 (20.0 hours)

- **Video Update:** [VLA_StealthSteering_PriyaA_Week7.mp4](../videos/VLA_StealthSteering_PriyaA_Week7.mp4) (Canonical Brutalist cream layout, Kokoro Bella voiceover, 1080p)
- **Problem Statement Review Document:** [`docs/RESEARCH_FRONTIERS_VERTICALS_REVIEW.md`](../docs/RESEARCH_FRONTIERS_VERTICALS_REVIEW.md) (Comprehensive evaluation of 12 open research verticals)
- **Experimental Plan & Threat Model:** [`docs/experiment_plan.md`](../docs/experiment_plan.md) (Formal formulation of Hypotheses 1 & 2 on OpenVLA-7B and LIBERO)
- **6-Week Execution Plan:** [`docs/weekly_execution_plan.md`](../docs/weekly_execution_plan.md) (Dual-engineer division of work across Attention Magnet and Auto-Feature Copy tracks)
- **Fellowship Renewal Agreement:** [`docs/renewal_agreement/Priya_Adlakha_HaiInc_Agreement_Renewal.pdf`](../docs/renewal_agreement/Priya_Adlakha_HaiInc_Agreement_Renewal.pdf) (Month 2 Fellowship Extension)

- **Daily Architecture Sync & Connects with Ameya (2.5 hrs).**
  Daily collaborative 3:00 PM PST sync calls (0.5 hr × 5 days, Monday through Friday) with Ameya Padwad to structure Month 2 fellowship research and coordinate technical handoffs:
  - Reviewed the 12 broad research verticals from the Humanitarians AI Research Frontiers initiative and aligned on pursuing the intersection of Physical AI Security (Vertical 5), Closed-Loop Evaluation (Vertical 1), and Simulation Testbeds (Vertical 2).
  - Divided Hypothesis 1 into two complementary, parallel research tracks to maximize empirical velocity:
    - *Track 1 (Priya Adlakha / Way 1):* "The Attention Magnet Patch" — targeting `Llama-2-7B` text-to-vision cross-attention layers, universal prompt-agnostic steering, and the differentiable `kornia` perspective warp on `LIBERO-Object`.
    - *Track 2 (Ameya Padwad / Way 2):* "The Auto-Feature Copy Patch" — targeting the `SigLIP + DINOv2` vision encoder and projector, automatic visual token matching, and native MuJoCo 3D coaster XML integration on `LIBERO-Spatial`.
  - Established a shared repository standard, Git branching protocols (`track1-attention-magnet` vs. `track2-feature-copy`), and weekly cross-validation checkpoints.

- **Research Frontiers Exploration & Problem Vertical Evaluation (5.5 hrs).**
  Dedicated extensive effort to evaluating open-ended research territories to finalize our fellowship project:
  - Conducted deep-dive literature reviews across all 12 broad research verticals: Intelligence Measurement, Learning Environments, Learning Algorithms, Cognitive Architectures, Security & Exploitation, Data & Knowledge, Compute & Execution, Formal Proof & Verification, Causal Science, Prediction & World Modeling, Inverse Design & Invention, and Sensing & Perception.
  - Investigated candidate problems in each vertical, evaluating theoretical novelty, physical relevance, compute feasibility (balancing local Apple Silicon prototyping with cloud A100 GPU pools), and alignment with Humanitarians AI’s mission.
  - Authored `docs/RESEARCH_FRONTIERS_VERTICALS_REVIEW.md`, summarizing candidate ideas, key trade-offs, and our formal justification for launching the Physical AI VLA Security & Robustness track.
  - Formulated the high-level intuition into accessible language:
    - *One-Liner:* "We are proving that a small tabletop sticker can quietly trick a robot into grabbing the wrong object by hijacking how its AI perceives the scene—and we are building a real-time safety shield that catches and stops this trick without needing to retrain the robot."
    - *Layperson Summary:* Breaking down the mechanics of Vision–Language–Action foundation models, explaining why high-level semantic hijacking circumvents classical motor-level vulnerabilities, and presenting the real-time Spatial Consistency Shield.

- **VLA Stealth Steering & Consistency Shield Experimental Design (5.5 hrs).**
  Authored the comprehensive, formal research design (`docs/experiment_plan.md`) combining two complementary hypotheses on OpenVLA-7B and LIBERO:
  - *Victim Architecture & Testbed Selection:* Selected official fine-tuned OpenVLA-7B checkpoints (`openvla-7b-finetuned-libero-object` and `openvla-7b-finetuned-libero-spatial`) on the Franka Emika Panda 7-DoF robot in MuJoCo, operating on $224 \times 224$ RGB camera frames and natural language commands.
  - *Hypothesis 1 (Semantic Object-Identity Swap):* Demonstrated the failure modes of classical motor-forcing patches (TMA-style) that freeze or twitch as camera perspectives shift; formulated the feature-swap loss operating on post-projector visual tokens ($16 \times 16$ grid) and language attention layers to steer the robot's closed-loop controller naturally toward Object B under benign instructions.
  - *Two-Stage 3D Placement Pipeline:* Separated patch optimization (differentiable homography perspective warp on 3D table plane) from evaluation rollouts (native MuJoCo rasterization on 3D surface geoms) to avoid simulation pixel cheating.
  - *Hypothesis 2 (Spatial Consistency Shield):* Designed a non-retrained runtime defense comparing Naive Random 5-View Masking against Systematic $3 \times 3$ Overlapping Grid Masking (extending *PatchCleanser* to continuous closed-loop control), combined with a Two-Round Minority Verification Check.

- **Technical Failure Mode Analysis & De-risking Formalization (4.0 hrs).**
  Conducted a rigorous audit of the experimental plan to identify subtle physical and algorithmic failure modes before execution, establishing five core de-risking protocols:
  1. *Differentiable-to-Native Render Transfer Gap:* Identified antialiasing, mipmapping, and specularity discrepancies between PyTorch homography warps and MuJoCo’s rasterizer; added light domain randomization (lighting/camera jitter) during patch training.
  2. *Spatial Token Coordinate Correspondence:* Diagnosed that fixed ViT grid indices cannot match between Object A and Object B due to absolute 2D positional embeddings; resolved via dynamic RoI bounding-box token pooling and Chamfer distance matching.
  3. *Real-Time 20 Hz Inference Bottleneck:* Calculated that evaluating 9 parallel 7B model forward passes per step drops control frequency to 2 Hz; formulated a dual-rate architecture running 20 Hz single-view control with lazy/anomaly-triggered shield verification.
  4. *Clean-Task False Positives from Occlusion:* Identified that masking the true target or the robot gripper spikes action variance on clean runs; established a kinematic arm exclusion zone and 3-step debounce threshold to guarantee $<5\%$ episode false alarms.
  5. *Physical Morphology & Grasp Affordance Compatibility:* Prevented collision and grasp slippage by pairing objects with compatible physical approach profiles (`black_bowl` vs. `ramekin`).

- **Brutalist Video Production & Fellowship Renewal Packaging (2.5 hrs).**
  Created the weekly video progress report and formalized fellowship extension documentation:
  - Authored the timecoded Brutalist video script (`video/video_script_week7.md`) adhering to canonical design standards: warm cream (`#F4EFEA`), espresso ink (`#2A241D`), and terracotta accent (`#D97757`).
  - Synthesized narration audio utilizing Kokoro TTS with the official Bella voice (`af_bella`) at 1.05× conversational pacing.
  - Rendered high-resolution presentation slides and compiled the 1080p MP4 video (`VLA_StealthSteering_PriyaA_Week7.mp4`).
  - Reviewed and packaged the fellowship renewal agreement (`Priya_Adlakha_HaiInc_Agreement_Renewal.pdf`) into the repository archives.

**Deliverables:**
- Master Problem Statement Review: `docs/RESEARCH_FRONTIERS_VERTICALS_REVIEW.md`
- Master Experiment Plan (with De-risking Protocols): `docs/experiment_plan.md`
- 6-Week Execution Schedule (Dual-Track Division): `docs/weekly_execution_plan.md`
- Week 7 Video Deliverable: `videos/VLA_StealthSteering_PriyaA_Week7.mp4`
- Dedicated Private Project Repository: `vla-physical-ai-stealth-shield`
- Executed Fellowship Renewal Agreement: `docs/renewal_agreement/Priya_Adlakha_HaiInc_Agreement_Renewal.pdf`
