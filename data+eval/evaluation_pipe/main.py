"""
Main entry point for the Mindvote  evaluation pipeline.
"""
import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

from .evaluator import PollEvaluator
from .config import MODELS, EVAL_CONFIG
from .utils import create_evaluation_dashboard, setup_logging
from .data_processing import get_data_statistics

logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables and directories."""
    # Create output directory
    Path(EVAL_CONFIG.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(
        log_level=EVAL_CONFIG.log_level,
        log_file=Path(EVAL_CONFIG.output_dir) / "evaluation.log"
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MindVote Poll Prediction Evaluation Pipeline')
    
    parser.add_argument(
        '--models', 
        nargs='+', 
        help='Models to evaluate (default: all available models)',
        choices=list(MODELS.keys()),
        default=None
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to evaluate',
        default=None
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results',
        default=EVAL_CONFIG.output_dir
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Data directory containing poll data',
        default=EVAL_CONFIG.data_dir
    )
    
    parser.add_argument(
        '--min-votes',
        type=int,
        help='Minimum number of votes for a poll to be included',
        default=100
    )
    
    parser.add_argument(
        '--max-options',
        type=int,
        help='Maximum number of options per poll',
        default=10
    )
    
    parser.add_argument(
        '--holdout-percentage',
        type=float,
        help='Percentage of options to hold out for prediction',
        default=0.3
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for processing',
        default=EVAL_CONFIG.batch_size
    )
    
    parser.add_argument(
        '--template',
        type=str,
        help='Prompt template to use',
        choices=['structured_template', 'minimal_template'],
        default='structured_template'
    )
    
    parser.add_argument(
        '--create-dashboard',
        action='store_true',
        help='Create evaluation dashboard'
    )
    
    parser.add_argument(
        '--skip-local',
        action='store_true',
        help='Skip local models (only use API models)'
    )
    
    parser.add_argument(
        '--skip-api',
        action='store_true',
        help='Skip API models (only use local models)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Prepare data and show statistics without running evaluation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def filter_models(args) -> List[str]:
    """Filter models based on command line arguments."""
    available_models = list(MODELS.keys())
    
    if args.models:
        selected_models = args.models
    else:
        selected_models = available_models
    
    # Filter based on model type
    if args.skip_local:
        selected_models = [m for m in selected_models if MODELS[m].type != 'local']
    
    if args.skip_api:
        selected_models = [m for m in selected_models if MODELS[m].type != 'api']
    
    return selected_models

def validate_environment(model_names: List[str]) -> List[str]:
    """Validate that required environment variables are set."""
    valid_models = []
    
    for model_name in model_names:
        if model_name not in MODELS:
            logger.warning(f"Model {model_name} not found in configuration")
            continue
        
        model_config = MODELS[model_name]
        
        if model_config.type == 'api':
            # Check if API key is available
            if not model_config.api_key:
                logger.warning(f"API key not found for {model_name}")
                continue
        
        valid_models.append(model_name)
    
    return valid_models

async def run_evaluation(args):
    """Run the evaluation pipeline."""
    # Setup environment
    setup_environment()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Filter and validate models
    selected_models = filter_models(args)
    valid_models = validate_environment(selected_models)
    
    if not valid_models:
        logger.error("No valid models found. Please check your configuration.")
        return
    
    logger.info(f"Starting evaluation with models: {valid_models}")
    
    # Create evaluator
    config = EVAL_CONFIG
    config.output_dir = args.output_dir
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    
    evaluator = PollEvaluator(config)
    
    # Prepare data
    logger.info("Preparing evaluation data...")
    samples = evaluator.prepare_data(
        min_votes=args.min_votes,
        max_options=args.max_options,
        holdout_percentage=args.holdout_percentage
    )
    
    # Show data statistics
    stats = get_data_statistics(samples)
    logger.info(f"Data statistics: {stats}")
    
    if args.dry_run:
        logger.info("Dry run completed. Exiting.")
        return
    
    # Limit samples if requested
    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")
    
    # Load models
    evaluator.load_models(valid_models)
    
    # Run evaluation
    logger.info("Starting model evaluation...")
    results = await evaluator.run_evaluation(
        model_names=valid_models,
        max_samples=args.max_samples
    )
    
    # Create dashboard if requested
    if args.create_dashboard:
        logger.info("Creating evaluation dashboard...")
        create_evaluation_dashboard(
            results['results'],
            results['metrics'],
            Path(args.output_dir) / "dashboard"
        )
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")

def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        asyncio.run(run_evaluation(args))
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 