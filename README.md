# Mindvote Evaluation Pipeline

## Overview

This pipeline evaluates Large Language Models (LLMs) on their ability to predict poll results by analyzing social contexts and user behavior patterns. The task involves predicting probability distributions over answer choices for polls from Reddit and Weibo platforms, with a current demonstration dataset of 1,400+ samples (200+ Reddit, 1,200+ Weibo) and demonstration metadata.

## Notice

The complete mindvote  dataset is currently undergoing validation to ensure data quality and ethical compliance. The demonstration samples are carefully curated to enable immediate reproducibility assessment while maintaining research integrity.

## Dataset


The complete dataset (3,918 polls) will be made publicly available following academic peer review.

## Features

- Multi-platform poll prediction evaluation (Reddit and Weibo)
- Support for both local and API-based language models
- Comprehensive metrics using 1-Wasserstein distance (Earth Mover's Distance)
- Automated evaluation dashboard generation
- Batch processing with customizable batch sizes
## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
- Set API keys for cloud-based models in config.py
- Adjust evaluation parameters in config.py as needed
- Ensure data files are placed in the specified data directory

## Usage

Basic evaluation:
```bash
python -m evaluation_pipe.main
```

Common options:
```bash
# Evaluate specific models
python -m evaluation_pipe.main --models gpt-4o llama-3-70b

# Generate evaluation dashboard
python -m evaluation_pipe.main --create-dashboard

# Skip API models (local only)
python -m evaluation_pipe.main --skip-api
```

## Evaluation Metrics

The pipeline uses the 1-Wasserstein distance (Earth Mover's Distance) to evaluate prediction quality:
- Measures the minimum "work" required to transform the predicted distribution into the ground truth
- Accounts for both the magnitude and position of probability mass differences
- Normalized to handle varying numbers of poll options

## Data Processing

The pipeline includes robust data processing features:
- Configurable holdout percentage for prediction evaluation
- Structured and minimal prompt templates
- Batch processing for efficient evaluation

## Output

The evaluation generates:
- Detailed metrics per model
- Aggregate performance comparisons
- Optional evaluation dashboard
- Comprehensive logging for analysis
- Raw prediction results for further analysis









