# Neural Semantic Role Labeling for IoT

This repository contains an initial scaffold for a neural Semantic Role Labeling (SRL) pipeline tailored to IoT applications.

## Project structure

- `data/raw/` - raw input data for preprocessing.
- `data/processed/` - processed data ready for model training and evaluation.
- `src/models/` - model definitions, training and inference code.
- `src/data_loader/` - data loading and preprocessing utilities.

## Setup

1. Create a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Initialize NLTK resources:

```bash
python init_environment.py
```

After this setup, you can extend the project with dataset ingestion, SRL model training, and IoT-specific deployment logic.
