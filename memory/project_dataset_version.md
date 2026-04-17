---
name: Dataset version mismatch
description: Local CICIoT2023 data has 40 columns (39 features + Label) instead of expected 47 columns (46 features + label) - different feature extraction version
type: project
---

Local data in data/ has 40-column version of CICIoT2023 (39 features + uppercase "Label"). The official/original version has 47 columns (46 features + lowercase "label"). Missing 9 columns: flow_duration, Duration, Srate, Drate, urg_count, Magnitue, Radius, Covariance, Weight. Has 2 extra: Time_To_Live, IGMP. The example.ipynb and thesis plan both assume 46 features.

**Why:** Two different feature extraction runs exist for the same PCAPs. The 40-col version is likely from Kaggle; the 47-col version is from the official UNB CIC page / IEEE DataPort.

**How to apply:** User needs to decide which dataset version to use before any implementation begins. All feature counts in the plan (input layer size, tensor shapes, feature extraction) depend on this decision.
