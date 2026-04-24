# Neural Semantic Role Labeling for IoT

This repository contains a full SRL pipeline for task-oriented IoT language understanding. It includes data preparation, a BERT-based baseline SRL model, a distillation/hyperparameter sweep pipeline, quantization and edge benchmarking, and an interactive command demo.

## Installation

### 1. Create and activate a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 4. Initialize NLP resources

```bash
python init_environment.py
```

This will download any required NLTK corpora used across preprocessing, predicate grounding, and demo utilities.

## Running the baseline SRL model

The baseline training entrypoint is `src/models/baseline_trainer.py`. It trains a token classification model on HuRIC-style BIO labels and writes a checkpoint to `outputs/baseline`.

```bash
python src/models/baseline_trainer.py
```

If you want to use a custom configuration file, edit `configs/baseline_config.yaml` and then run it directly.

When training completes, the saved model, tokenizer, config and evaluation metrics are stored in:

- `outputs/baseline/`

## Running the hyperparameter sweep and distilled student

The sweep script searches for a compact student model configuration and saves the best trial under `outputs/sweep_trials/best_student`.

```bash
python src/models/run_sweep.py
```

This script uses:

- `outputs/augmented` as the teacher model
- the augmented HuRIC dataset in `data/processed/huric_augmented`
- `configs/baseline_config.yaml` for base hyperparameters

After the sweep finishes, the best student model is also copied to:

- `outputs/best_student/`

## Running the interactive demo

Use the interactive demo to type natural language IoT commands and inspect SRL outputs.

```bash
python src/demo_iot.py
```

During demo execution, the script loads the teacher model from `outputs/baseline` by default and prints raw token-to-label mappings before producing final JSON output.

## Optional utilities

- `bash sync_to_colab.sh` — creates a zip containing selected model checkpoints and automates Git staging/commit/push for Colab transfer.
- `src/analysis/iot_benchmark.py` — benchmark script for comparing baseline, student, and quantized models on size/latency/F1.
- `src/models/quantize_model.py` and `src/models/quantizer.py` — scripts for converting the student model to 4-bit quantized weights.

## Folder structure

```text
.
├── configs/                    # YAML config files for training and sweeps
├── data/
│   ├── raw/                    # Raw source data for preprocessing
│   └── processed/              # Tokenized HuRIC datasets ready for training
├── outputs/                    # Saved model checkpoints, metrics, and artifacts
├── src/
│   ├── analysis/               # Evaluation, benchmarking, and plotting utilities
│   ├── data_loader/            # Dataset parsing, token alignment, and preprocessing helpers
│   └── models/                 # Training scripts, distillation logic, and quantization helpers
├── .gitignore                  # Ignore rules for temporary files and large outputs
├── init_environment.py         # NLTK / environment bootstrap script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── sync_to_colab.sh            # Colab sync helper script
```

## Notes

- Keep large binary files out of Git by using the `.gitignore` entries for `outputs/` and `*.zip`.
- If you add new model directories or temporary checkpoints, update `sync_to_colab.sh` accordingly.
- Use the interactive demo after you have trained or downloaded the baseline model to verify predicate extraction and label alignment.
