# CIC-IoT-2023 IDS Literature — Research Questions, Motivations & Gaps

> **Purpose.** Source material for the **Introduction** and **Theoretical Background / Related Work** chapters of the bachelor thesis (real-time MLP-based IDS on CIC-IoT-2023). For each notable paper this captures *the research questions, problem statements, motivations, and gaps the authors themselves articulate* — not their results. Cross-cutting themes are grouped so they map onto thesis chapters.
>
> **Method.** Deep-research harness: 5 search angles → 23 sources fetched → 109 candidate claims → 25 adversarially verified (3-vote, 2/3 to kill) → 22 confirmed, 3 refuted. Generated 2026-06-14.
>
> **How to read confidence.** `high` = verified against primary full-text or an open-access mirror; `medium` = verified at abstract level or the canonical source is off-dataset/paywalled. Refuted claims are listed at the end so they are not accidentally cited.

---

## TL;DR — the recurring research questions

Across the CIC-IoT-2023 IDS literature, authors motivate their work through a small, repeating set of questions that line up almost one-to-one with the thesis chapters:

1. **Class imbalance** — the single most universal stated gap: benign/majority traffic swamps rare attack classes, biasing detectors and degrading minority-class recall.
2. **The accuracy ↔ computational-cost ↔ real-time triad** — existing IoT IDS allegedly fail to satisfy all three jointly on resource-constrained, evolving IoT.
3. **Classification granularity** — binary (benign vs malicious) vs multiclass (attack family / variant) is an explicit axis of variation.
4. **Deep learning vs classical/tree ensembles** — including hybrid DL+ML (FFNN+XGBoost) and spatial+temporal hybrids (AE-LSTM-CNN); framed as "standalone models can't do X."
5. **Generalization / concept drift** — heterogeneous, evolving IoT causes distribution shift; static models trained on fixed datasets go stale.
6. **Explainability / XAI** — countering the black-box opacity of DL IDS for transparency and trust.

The originating dataset paper (Neto et al., *Sensors* 2023) frames its *own* motivation as the inadequacy of prior IoT datasets, and defines the **2-/8-/34-class taxonomy** the thesis uses. A 2026 Springer survey corroborates these six themes as a coherent challenge taxonomy and confirms CIC-IoT-2023 as a current benchmark whose scale itself challenges lightweight training.

> ⚠️ **Citation hazard.** Several frequently-surfaced IoT-IDS papers (Sadhwani 2025 XAI; the TON_IoT XAI paper; IoT-23 and RT-IoT2022 studies; NSL-KDD resampling) **do NOT actually use CIC-IoT-2023**. Cite them only for cross-cutting themes, never as CIC-IoT-2023 studies. See [§4 Off-scope](#4-off-scope-papers--do-not-cite-as-cic-iot-2023-studies).

---

## 1. The originating dataset paper

### Neto et al. (2023) — *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment*
- **Venue:** *Sensors* 23(13):5941 (MDPI), peer-reviewed primary. — **confidence: high**
- **Stated goal (verbatim):** *"The main goal of this research is to propose a novel and extensive IoT attack dataset to foster the development of security analytics applications in real IoT operations."*
- **Stated gap (verbatim):** *"Most existing efforts do not consider an extensive network topology with real IoT devices."*
- **Taxonomy it establishes (defines the thesis granularities):** 33 attacks executed by/against **105 real IoT devices**, grouped into **7 families: DDoS, DoS, Recon, Web-based, Brute Force, Spoofing, Mirai.**
  - **2-class** = Benign vs Attack
  - **8-class** = 7 families + Benign
  - **34-class** = 33 variants + Benign
- **Links:**
  - Publisher: https://www.mdpi.com/1424-8220/23/13/5941
  - Open-access mirror (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/
  - Official dataset page (UNB CIC): https://www.unb.ca/cic/datasets/iotdataset-2023.html

> **Refuted sub-claim (do not cite):** that the dataset paper frames its *core* task as binary 2-class detection. Verification killed this 1–2 — the paper benchmarks both ML and DL across the full taxonomy, not a binary-first framing. The 2-class lens is the *thesis's* deployment framing, not the dataset paper's.

---

## 2. Cross-cutting research questions (grouped by theme)

These are the reusable "the literature says…" paragraphs for the Theoretical Background. Each notes how the thesis relates.

### 2.1 Class imbalance — *the* most common stated gap — **confidence: high**
- **2026 Springer survey:** *"a persistent issue in ML-based IDS … benign traffic significantly outweighs attack samples, leading to biased models that underperform in detecting rare or sophisticated threats."*
- **AegisGuard (Sensors 2025, uses CIC-IoT-2023):** *"One of the most persistent challenges … causing many frameworks to overlook rare but critical attacks"* — addressed with a four-stage SMOTE / SMOTEENN / ADASYN / under-sampling pipeline.
- **Susilo et al. (Sensors 2025, CIC-IoT-2023):** lists it among co-equal gaps — *"existing methods often fail to fully address critical challenges, such as class imbalance, temporal feature extraction, and the integration of static and dynamic data patterns"* — mitigated with SMOTE.
- **Alashjaee & Alqahtani / FFNN+XGBoost (Sci Rep 2025, CIC-IoT-2023):** *"class imbalance issues persist unaddressed, leading to biased detection."*
- **→ Thesis relation:** the literature overwhelmingly uses **SMOTE-family oversampling**. The thesis uses **per-class cap + class-weighted cross-entropy (no SMOTE)** — a deliberate, defensible deviation that should be *explicitly justified against this norm* (memory + meaningless synthetic flows). Frame as a methodological choice, not an omission.
- Sources: https://link.springer.com/article/10.1007/s42452-026-08458-8 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12655908/ · https://pmc.ncbi.nlm.nih.gov/articles/PMC11768945/ · https://www.nature.com/articles/s41598-025-20047-0

### 2.2 Accuracy ↔ computational cost ↔ real-time efficiency triad — **confidence: high**
- **Alashjaee & Alqahtani (Sci Rep 2025, CIC-IoT-2023), abstract:** *"Traditional Intrusion Detection Systems (IDS) struggle to detect sophisticated attacks in real-time due to resource constraints and evolving attack patterns."*
- Same paper, Literature Review (verbatim gap): *"existing models face challenges including high computational complexity, low detection accuracy, and inefficiency in working with real-time attacks"* → *"This research presents a hybrid model that overcomes these gaps."*
- **→ Thesis relation / caveat:** the literature's "real-time" is the **broad** framing (line-rate, resource-constrained deployment). The thesis scope is narrower — **interactive single-request inference (<100 ms end-to-end)**. Cite these for *motivation*, but **do not let cited latency claims imply line-rate parity** with the thesis demo.
- Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC12528720/ · https://www.nature.com/articles/s41598-025-20047-0

### 2.3 Deep learning vs classical/tree ensembles (and hybrid fusion) — **confidence: high**
- **Alashjaee & Alqahtani (CIC-IoT-2023):** gap = *"Most existing works do not effectively integrate DL and ML for improved performance"* → proposes hybrid **FFNN (deep feature extraction) + XGBoost (classifier).**
- **Susilo et al. (CIC-IoT-2023, Sensors 2025):** *"Conventional approaches typically employ standalone models, which lack the capability to simultaneously process spatial and temporal dimensions"* → motivates a hybrid **AE-LSTM-CNN.**
- **→ Thesis relation / nuance:** these papers frame it as **hybrid-DL vs standalone-DL**, *not* strictly DL vs tree baselines. The thesis's **MLP-vs-RandomForest** head-to-head on identical splits addresses the committee's "does a tree beat the MLP?" question **more directly than the literature does.**
- Sources: https://www.nature.com/articles/s41598-025-20047-0 · https://pmc.ncbi.nlm.nih.gov/articles/PMC11768945/

### 2.4 Trees are a strong baseline on CIC-IoT-2023 — **confidence: high (lower-tier venue)**
- **Jony & Arnob (IJITCS v16 n4, 2024, CIC-IoT-2023, binary):** Decision Tree **0.9919** and Random Forest **0.9916** accuracy vs KNN 0.9380 and Logistic Regression 0.8275; concludes *"DT and RF are strong contenders in the field of IoT intrusion detection."*
- **Caveats:** MECS-Press IJITCS is a **lower-tier journal**; DT is a single tree (only RF is an ensemble); DL hybrids sometimes match/slightly exceed trees. Use as **one supporting data point**, not a universal finding.
- **→ Thesis relation:** directly supports including a RandomForest baseline to pre-empt the tree-vs-MLP committee question.
- Sources: https://www.mecs-press.org/ijitcs/ijitcs-v16-n4/v16n4-4.html · https://www.mecs-press.org/ijitcs/ijitcs-v16-n4/IJITCS-V16-N4-4.pdf

### 2.5 Classification granularity (binary vs multiclass) — **confidence: high**
- **Binary-only camp:** Jony & Arnob frame the RQ as comparing four classical ML models (LR, KNN, DT, RF) *"to determine their effectiveness in detecting and preventing cyber threats,"* operating strictly binary (they under-sample *"malicious packets to the same level as … legitimate packets,"* all metrics binary TP/TN/FP/FN, no 8-/34-class experiment).
- **Multiclass camp:** Susilo (AE-LSTM-CNN), FFNN+XGBoost, and AegisGuard target attack families/variants.
- **→ Thesis relation:** the *split itself* between binary-only and multiclass papers is the granularity research question — which **validates the thesis's 2-/8-/34-class comparison as a recognized contribution.**
- Sources: https://www.mecs-press.org/ijitcs/ijitcs-v16-n4/v16n4-4.html

### 2.6 Generalization & concept drift — **confidence: high (theme named, rarely tested)**
- **2026 Springer survey** names a distinct challenge category: *"Heterogeneity and Concept Drift: IoT networks are highly heterogeneous and dynamic, causing frequent shifts in data distributions that degrade model performance over time,"* adding *"Static models trained on fixed datasets may become obsolete as threats evolve."*
- **AegisGuard (CIC-IoT-2023):** existing approaches *"achieve high accuracy on specific datasets but lack generalizability, interpretability, and stability when deployed across heterogeneous IIoT environments"* → motivates four-dataset evaluation.
- **→ Thesis relation (strong novelty angle):** these papers **name** concept drift but **do not perform temporal/drift evaluation.** The thesis's **temporal split (train-on-past/test-on-future)** is a *stronger, less-common empirical treatment* than the survey norm — frame the expected F1 gap as the honest generalization finding, not a failure.
- Sources: https://link.springer.com/article/10.1007/s42452-026-08458-8 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12655908/

### 2.7 Explainability / XAI — **confidence: medium (canonical source is off-dataset)**
- **Sadhwani et al. (Computers & Electrical Engineering 2025):** gap = ML/DL models *"have a black-box nature and lack interpretability, and explainable artificial intelligence works towards improving the model's transparency and trustworthiness"* (applies SHAP/LIME).
- **AegisGuard** also lists interpretability among its gaps.
- **⚠️ Important:** Sadhwani et al. **do NOT use CIC-IoT-2023** (they use NSL-KDD, UNSW-NB15, TON-IoT, X-IIoTID). Cite as **general XAI-IDS motivation only.** Confidence is medium because the canonical XAI source is off-dataset and the full text was paywalled (verified via abstract + ACM mirror).
- **→ Thesis relation:** supports the thesis's SHAP-on-alert demo feature as addressing a recognized gap.
- Sources: https://www.sciencedirect.com/science/article/abs/pii/S0045790625001995 · https://dl.acm.org/doi/10.1016/j.compeleceng.2025.110256 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12655908/

### 2.8 CIC-IoT-2023 scale as a stated challenge — **confidence: high**
- **2026 Springer survey:** *"CICIoT2023 is one of the most comprehensive, with over 100 devices and 34 attack classes, though its size and complexity pose challenges for training lightweight models."*
- **→ Thesis relation:** justifies the **sampling / per-class-cap** decisions in Theoretical Background — the dataset's scale is itself a cited reason to subsample.
- Minor caveat: survey stats ("34 attack classes / over 100 devices") are loose vs the official 33 attacks / 105 devices / 34 classes-with-Benign.
- Source: https://link.springer.com/article/10.1007/s42452-026-08458-8

---

## 3. Per-paper quick reference (CIC-IoT-2023 studies only)

| Paper | Venue / year | Stated RQ / gap (their framing) | Granularity | Imbalance method | Confidence | Link |
|---|---|---|---|---|---|---|
| **Neto et al. — CICIoT2023** | Sensors 2023 | Prior IoT datasets inadequate; build large-scale real-device benchmark | Defines 2/8/34 | (dataset, n/a) | high | [MDPI](https://www.mdpi.com/1424-8220/23/13/5941) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/) |
| **Alashjaee & Alqahtani — FFNN+XGBoost** | Sci Rep 2025 | DL and ML not effectively integrated; accuracy/complexity/real-time gaps | Multiclass | (states gap) | high | [Nature](https://www.nature.com/articles/s41598-025-20047-0) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12528720/) |
| **Susilo et al. — AE-LSTM-CNN** | Sensors 2025 | Standalone models can't capture spatial+temporal jointly; imbalance; static/dynamic fusion | Multiclass | SMOTE | high | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11768945/) |
| **AegisGuard** | Sensors 2025 | Imbalance; lack of generalizability/interpretability/stability across IIoT | Multiclass | 4-stage SMOTE/SMOTEENN/ADASYN/undersample | high | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12655908/) |
| **Jony & Arnob — classical ML** | IJITCS 2024 | Compare LR/KNN/DT/RF effectiveness for IoT threat detection | Binary | undersampling | high (low-tier venue) | [HTML](https://www.mecs-press.org/ijitcs/ijitcs-v16-n4/v16n4-4.html) · [PDF](https://www.mecs-press.org/ijitcs/ijitcs-v16-n4/IJITCS-V16-N4-4.pdf) |
| **2026 IoT-IDS survey** | Discover Applied Sciences 2026 | Synthesizes challenge taxonomy (imbalance, drift, scale, XAI) | survey | survey | high | [Springer](https://link.springer.com/article/10.1007/s42452-026-08458-8) |

---

## 4. Off-scope papers — DO NOT cite as CIC-IoT-2023 studies

Verified (3-0) that these do **not** train on CIC-IoT-2023, despite surfacing in searches. Cite only for the cross-cutting *theme* noted, never as a CIC-IoT-2023 result. — **confidence: high**

| Paper | Actual dataset(s) | Safe to cite for | Link |
|---|---|---|---|
| Sadhwani et al. 2025 (XAI) | NSL-KDD, UNSW-NB15, TON-IoT, X-IIoTID | XAI/black-box motivation | https://www.sciencedirect.com/science/article/abs/pii/S0045790625001995 |
| XAI over IoT data streams | TON_IoT (network + 6 sensor sets) | XAI; CIC-IoT-2023 only cited in related work | https://pmc.ncbi.nlm.nih.gov/articles/PMC11820747/ |
| Multiclass real-time IDS | **IoT-23** (Stratosphere Lab 2020 — different dataset, similar name) | real-time multiclass framing | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11281211/ |
| Imbalanced-data IDS (arXiv 2505.10600) | **RT-IoT2022** (UCI; 123,117 inst, 83 feat, 12 cls) | imbalance; CIC-IoT only cited ref [9] | https://arxiv.org/pdf/2505.10600 |
| Abdelkhalek & Mashaly 2023 | NSL-KDD only | class-imbalance resampling theme | https://link.springer.com/article/10.1007/s11227-023-05073-x |

---

## 5. Claims that were REFUTED in verification (excluded — do not cite)

1. **(1–2)** That the CICIoT2023 dataset paper frames its core task as binary classification. → It benchmarks ML *and* DL across the full taxonomy. Source: https://www.mdpi.com/1424-8220/23/13/5941
2. **(0–3)** That arXiv:2505.10600's central problem is a specific 94,659-vs-28 majority/minority imbalance solved by hybrid sampling. → Not its framing (and that paper is off-dataset anyway). Source: https://arxiv.org/pdf/2505.10600
3. **(0–3)** That AegisGuard uses quantum-inspired feature selection reducing CIC-IoT-2023 from 44→12 features (70.6%). → Not supported. Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12655908/

---

## 6. Caveats on the evidence base

- **Source strength varies.** Strong peer-reviewed primaries: the dataset paper (Sensors 2023), FFNN+XGBoost (Sci Rep 2025), Susilo AE-LSTM-CNN (Sensors 2025), AegisGuard (Sensors 2025), the 2026 Springer survey. **Lower-tier:** Jony & Arnob (MECS-Press IJITCS) — use as a supporting data point only.
- **Fetch limitations.** Several MDPI/ScienceDirect/ACM pages returned HTTP 403; verification often relied on open-access PMC mirrors and exact-match abstract searches rather than publisher full-text. For paywalled papers (notably Sadhwani et al.) verification is **abstract-level**.
- **Time-sensitivity.** The most relevant CIC-IoT-2023 papers cluster in **2024–2026**; the field is actively growing and this snapshot may miss very recent work.
- **Two framing nuances to protect in the thesis:**
  - (a) The literature's "real-time" is **broader** than the thesis's interactive-single-request scope — don't let cited latency motivations imply line-rate parity.
  - (b) Papers **name** concept drift / generalization but rarely run temporal-split evaluation → the thesis's temporal split is comparatively **novel**, not standard.
- **Imbalance norm.** Class-imbalance papers overwhelmingly use SMOTE-family oversampling; the thesis's class-weighted-CE-only approach is a deliberate deviation that should be explicitly defended.

---

## 7. Open questions worth chasing for the thesis

1. Which CIC-IoT-2023 papers (if any) *actually perform* temporal/concept-drift evaluation rather than just naming it? Few direct comparators → strengthens novelty but weakens benchmark comparability.
2. What macro-F1 / per-class numbers do the strong primaries (FFNN+XGBoost, AE-LSTM-CNN, AegisGuard) report **on the 8-class and 34-class** granularities specifically, so the thesis can position its MLP on a matching taxonomy?
3. Do any CIC-IoT-2023 papers use the same **39-feature CSV release** the thesis uses, or do they all use the 46-feature version? Affects direct comparability of reported numbers.
4. Beyond Jony & Arnob, is there a peer-reviewed CIC-IoT-2023 study with a head-to-head **MLP-vs-RandomForest/XGBoost** comparison on identical splits, to most directly pre-empt the committee's tree-vs-MLP question?

---

## 8. Full source list (as fetched)

**Confirmed CIC-IoT-2023 + survey primaries**
- Neto et al., CICIoT2023 — https://www.mdpi.com/1424-8220/23/13/5941 · https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/ · https://www.unb.ca/cic/datasets/iotdataset-2023.html
- FFNN+XGBoost (Alashjaee & Alqahtani) — https://www.nature.com/articles/s41598-025-20047-0 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12528720/
- AE-LSTM-CNN (Susilo et al.) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11768945/
- AegisGuard — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12655908/
- Jony & Arnob (classical ML) — https://www.mecs-press.org/ijitcs/ijitcs-v16-n4/v16n4-4.html · https://www.mecs-press.org/ijitcs/ijitcs-v16-n4/IJITCS-V16-N4-4.pdf
- 2026 IoT-IDS survey — https://link.springer.com/article/10.1007/s42452-026-08458-8

**Off-scope (theme-only citations)**
- Sadhwani et al. (XAI) — https://www.sciencedirect.com/science/article/abs/pii/S0045790625001995 · https://dl.acm.org/doi/10.1016/j.compeleceng.2025.110256
- XAI over IoT streams (TON_IoT) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11820747/
- Multiclass real-time (IoT-23) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11281211/
- Imbalanced data (RT-IoT2022) — https://arxiv.org/pdf/2505.10600
- Abdelkhalek & Mashaly (NSL-KDD) — https://link.springer.com/article/10.1007/s11227-023-05073-x

**Other sources fetched (not individually cited above; mixed relevance)**
- https://arxiv.org/pdf/2502.06031 · https://www.nature.com/articles/s41598-025-23711-7 · https://www.researchgate.net/publication/396085029 · https://arxiv.org/pdf/2512.02272 · https://link.springer.com/article/10.1007/s43926-025-00203-8 · https://pmc.ncbi.nlm.nih.gov/articles/PMC10557502/ · https://www.mdpi.com/2571-5577/8/2/52 · https://arxiv.org/abs/2603.22771 · https://link.springer.com/article/10.1007/s12243-025-01118-9 · https://www.mdpi.com/1999-5903/16/8/284

---

*Run stats: 5 angles · 23 sources fetched · 109 claims extracted · 25 verified · 22 confirmed / 3 killed · 105 agents.*
