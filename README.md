# cripts-elfen_adapter.py
# Cross-Platform Blind Spot: MLRan vs. Linux Ransomware

This repository contains the code, extracted feature data, and simulation results for the paper **"Does a Windows Ransomware Dataset Describe Linux Ransomware? The Cross-Platform Blind Spot of MLRan"**.

## Contents

- `scripts/elfen_adapter.py`: Python script to extract the Linux-schema features (Section 5) from ELFEN sandbox JSON reports.
- `scripts/sparsity_sim.py`: Python script for the feature sparsity simulation (Section 4.5) on MLRan's test set.
- `scripts/replot.py`: Script to generate Figure 4 (sparsity plot).
- `data/features_all.jsonl`: The extracted features from the 32 ELFEN reports across 16 families (Table 7).
- `data/RFE_selected_feature_names_dic.json`: The dictionary of the 483 selected features from MLRan, mapping indices to Cuckoo vocabulary.
- `results/`: CSV outputs of the sparsity simulation.

## Reproduction

### 1. ELFEN Feature Extraction (Table 7)
To extract features from ELFEN reports, you will need the ELFEN dataset (DOI: 10.5281/zenodo.19617322).
Run:
```bash
python scripts/elfen_adapter.py /path/to/ELFEN_reports/ --mlran-rfe data/RFE_selected_feature_names_dic.json --output data/features_all.jsonl
