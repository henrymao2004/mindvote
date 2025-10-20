# Mindvote Evaluation Pipeline

## Overview

This pipeline evaluates Large Language Models (LLMs) on their ability to predict poll results by analyzing social contexts and user behavior patterns. The task involves predicting probability distributions over answer choices for polls from Reddit and Weibo platforms.

## Notice

The mindvote  dataset is currently undergoing validation to ensure data quality and ethical compliance. 


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

The pipeline uses multiple complementary metrics to evaluate prediction quality, following the OpinionQA benchmark methodology:

### 1-Wasserstein Distance (Earth Mover's Distance)
- Measures the minimum "work" required to transform the predicted distribution into the ground truth
- Accounts for both the magnitude and position of probability mass differences
- Normalized to handle varying numbers of poll options
- Lower values indicate better performance

### KL Divergence (Kullback-Leibler Divergence)
- Measures how different the predicted probability distribution is from the ground truth
- Sensitive to exact probability values and penalizes confident wrong predictions
- Lower values indicate better performance
- Asymmetric measure: KL(P||Q) ≠ KL(Q||P)

### Spearman Rank Correlation
- Measures the correlation between the rankings of predicted and ground truth distributions
- Evaluates whether the model captures the relative ordering of options correctly
- Range: [-1, 1], where 1 indicates perfect positive correlation
- Higher values indicate better performance

### One-Hot Accuracy (Top-1 Accuracy)
- Measures how often the model's top choice matches the ground truth's top choice
- Simple binary accuracy metric for the most probable option
- Range: [0, 1], where 1 indicates perfect accuracy
- Higher values indicate better performance




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









