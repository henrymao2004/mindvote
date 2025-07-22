"""
Configuration settings for the MindVote evaluation pipeline.
"""
import os
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    type: str  # 'api' or 'local'
    model_path: str = ""
    api_key: str = ""
    base_url: str = ""
    max_concurrent_requests: int = 10
    rate_limit_per_minute: int = 60
    timeout: int = 300
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 1000

@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    data_dir: str = "data"
    output_dir: str = "results"
    batch_size: int = 32
    max_workers: int = 8
    enable_reasoning: bool = True
    save_intermediate: bool = True
    log_level: str = "INFO"

# Model configurations
MODELS = {
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        type="api",
        model_path="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url="https://api.openai.com/v1",
        max_concurrent_requests=8,
        rate_limit_per_minute=100,
        timeout=300,
        max_retries=3,
        temperature=0.0,
        max_tokens=1000
    ),
    "gpt-4.1": ModelConfig(
        name="gpt-4.1",
        type="api",
        model_path="gpt-4.1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url="https://api.openai.com/v1",
        max_concurrent_requests=8,
        rate_limit_per_minute=100,
        timeout=300,
        max_retries=3,
        temperature=0.0,
        max_tokens=1000
    ),
    "gemini-2.5-pro": ModelConfig(
        name="gemini-2.5-pro",
        type="api",
        model_path="gemini-2.5-pro",
        api_key=os.getenv("GOOGLE_API_KEY", ""),
        base_url="https://generativelanguage.googleapis.com/v1beta",
        max_concurrent_requests=6,
        rate_limit_per_minute=60,
        timeout=300,
        max_retries=3,
        temperature=0.0,
        max_tokens=1000
    ),
    "deepseek-r1": ModelConfig(
        name="deepseek-r1",
        type="api",
        model_path="deepseek-r1",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com/v1",
        max_concurrent_requests=8,
        rate_limit_per_minute=100,
        timeout=300,
        max_retries=3,
        temperature=0.0,
        max_tokens=1000
    ),
    
    # Local Models
    "llama-3-70b": ModelConfig(
        name="llama-3-70b",
        type="local",
        model_path="meta-llama/Llama-3-70B-Instruct",
        max_concurrent_requests=4,
        timeout=300,
        max_retries=1,
        temperature=0.0,
        max_tokens=1000
    ),
    "gemma-2-27b": ModelConfig(
        name="gemma-2-27b",
        type="local",
        model_path="google/gemma-2-27b-it",
        max_concurrent_requests=4,
        timeout=300,
        max_retries=1,
        temperature=0.0,
        max_tokens=1000
    ),
    "qwen-2.5-72b": ModelConfig(
        name="qwen-2.5-72b",
        type="local",
        model_path="Qwen/Qwen2.5-72B-Instruct",
        max_concurrent_requests=4,
        timeout=300,
        max_retries=1,
        temperature=0.0,
        max_tokens=1000
    ),
}

# Evaluation configuration
EVAL_CONFIG = EvaluationConfig(
    data_dir="data",
    output_dir="results",
    batch_size=32,
    max_workers=8,
    enable_reasoning=True,
    save_intermediate=True,
    log_level="INFO"
)

# Prompt templates
PROMPT_TEMPLATES = {
    "structured_template": """You are a poll prediction expert analyzing voting patterns and social dynamics. Your task is to predict the percentage distribution of poll responses based on the provided context.

**Input Processing:**
- You will receive a JSON object containing poll information
- Some options will have "percentage": "Guess Here" - these are the ones you need to predict
- Other options may already have known percentages that you should preserve

**Output Constraints:**
- Each percentage must be an integer or have one decimal place
- Do not include the % symbol
- All percentages must sum to exactly 100 (±0.1 rounding error allowed)
- Do not infer vote counts from sample size

**Return Format:**
- Return exactly one JSON block
- Use identical schema to input
- Replace all "Guess Here" fields with your numeric predictions
- Preserve all key names, spacing, and order
- No extra keys, comments, or explanatory text outside the JSON

**Poll Information:**
{poll_data}

**Context:**
{context}

**Step-by-Step Reasoning:**
Please provide your reasoning process before giving the final JSON output.

**Final JSON Output:**""",

    "minimal_template": """Predict the missing percentages in this poll. Replace "Guess Here" with numbers. All percentages must sum to 100.

{poll_data}

Context: {context}

Return only the JSON with your predictions:"""
}

# Evaluation metrics to compute
METRICS = [
    "wasserstein_1", # 1-Wasserstein distance (Earth Mover's Distance)
]

# Data processing settings
DATA_SETTINGS = {
    "min_total_votes": 100,      # Minimum votes for a poll to be included
    "max_options": 8,           # Maximum number of options per poll
    "test_split": 0.2,           # Fraction of data for testing
    "random_seed": 42,           # Random seed for reproducibility
    "holdout_percentage": 0.3,   # Percentage of options to hold out for prediction
} 