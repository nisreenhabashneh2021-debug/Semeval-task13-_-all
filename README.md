# Semeval-task13-_-all
Code for SemEval-2026 Task # SemEval-2026 Task 13 — Full System (Subtasks A, B, C)

This repository contains our systems for **all three subtasks** of SemEval-2026
Task 13 (Code Authorship Detection). We implement:

- Classical baselines (TF-IDF + Logistic Regression)
- Transformer models (BERT, CodeBERT, CodeT5-small)
- Simple ensembles

The code is organized so that each subtask can be run independently, while
sharing common utilities for data loading and evaluation.

---

## Repository Structure

```text
Semeval-task13-_-all/
├── common/          # shared utilities (data loading, metrics, plotting)
├── Sub Task A/      # Subtask A system
├── Sub Task B/      # Subtask B system
├── Sub Task C/      # Subtask C system
├── notebooks/       # optional EDA / error analysis
├── scripts/         # helper scripts (e.g., data download)
├── requirements.txt
└── README.md
13
