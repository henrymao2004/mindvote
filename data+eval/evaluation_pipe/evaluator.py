"""
Main evaluator for the MindVote poll prediction pipeline.
"""
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
from dataclasses import dataclass, asdict
import time

from .config import MODELS, EVAL_CONFIG, PROMPT_TEMPLATES
from .data_processing import DataProcessor, EvaluationSample
from .models import ModelFactory, BaseModel
from .metrics import PollMetrics, compute_aggregate_metrics

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    model_name: str
    sample_id: str
    prediction: Dict[str, Any]
    ground_truth: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    response_time: float = 0.0
    raw_response: str = ""

class PollEvaluator:
    """Main evaluator for poll prediction models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or EVAL_CONFIG
        self.data_processor = DataProcessor(
            data_dir=self.config.data_dir,
            random_seed=42
        )
        self.models = {}
        self.metrics = PollMetrics()
        self.results = []
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.config.output_dir) / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_models(self, model_names: List[str] = None):
        """Load specified models."""
        if model_names is None:
            model_names = list(MODELS.keys())
        
        model_configs = {name: asdict(MODELS[name]) for name in model_names if name in MODELS}
        self.models = ModelFactory.create_models(model_configs)
        
        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def prepare_data(self, min_votes: int = 100, max_options: int = 10, 
                    holdout_percentage: float = 0.3) -> List[EvaluationSample]:
        """Prepare evaluation data."""
        logger.info("Loading and preparing data...")
        
        # Load poll data
        polls = self.data_processor.load_poll_data()
        
        # Filter polls
        polls = self.data_processor.filter_polls(
            polls, min_votes=min_votes, max_options=max_options
        )
        
        # Create evaluation samples
        samples = self.data_processor.create_evaluation_samples(
            polls, holdout_percentage=holdout_percentage
        )
        
        logger.info(f"Prepared {len(samples)} evaluation samples")
        return samples
    
    def create_prompt(self, sample: EvaluationSample, template_name: str = "structured_template") -> str:
        """Create prompt for a sample."""
        template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["structured_template"])
        
        poll_data = json.dumps(sample.input_data, indent=2)
        context = sample.context
        
        prompt = template.format(
            poll_data=poll_data,
            context=context
        )
        
        return prompt
    
    def parse_prediction(self, response: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse model response to extract prediction."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Look for JSON block without markdown
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Try to extract from the end of response
                    lines = response.strip().split('\n')
                    for i in range(len(lines) - 1, -1, -1):
                        if lines[i].strip().startswith('{'):
                            json_str = '\n'.join(lines[i:])
                            break
                    else:
                        raise ValueError("No JSON found in response")
            
            # Parse JSON
            prediction = json.loads(json_str)
            
            # Validate prediction has required structure
            if 'options' not in prediction:
                raise ValueError("Missing 'options' field in prediction")
            
            # Ensure all "Guess Here" fields are replaced
            for option in prediction['options']:
                if isinstance(option.get('percentage'), str) and 'guess here' in option['percentage'].lower():
                    raise ValueError("Not all 'Guess Here' fields were replaced")
            
            return prediction
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error: {str(e)}")
            # Return fallback prediction
            return self._create_fallback_prediction(input_data)
        except Exception as e:
            logger.warning(f"Prediction parsing error: {str(e)}")
            return self._create_fallback_prediction(input_data)
    
    def _create_fallback_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback prediction when parsing fails."""
        prediction = input_data.copy()
        
        # Count "Guess Here" fields
        guess_here_count = 0
        for option in prediction['options']:
            if isinstance(option.get('percentage'), str) and 'guess here' in option['percentage'].lower():
                guess_here_count += 1
        
        if guess_here_count == 0:
            return prediction
        
        # Calculate known percentages
        known_total = 0
        for option in prediction['options']:
            if isinstance(option.get('percentage'), (int, float)):
                known_total += option['percentage']
        
        # Distribute remaining percentage equally
        remaining = max(0, 100 - known_total)
        equal_share = remaining / guess_here_count if guess_here_count > 0 else 0
        
        # Replace "Guess Here" with equal shares
        for option in prediction['options']:
            if isinstance(option.get('percentage'), str) and 'guess here' in option['percentage'].lower():
                option['percentage'] = round(equal_share, 1)
        
        return prediction
    
    async def evaluate_sample(self, model: BaseModel, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a single sample with a model."""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self.create_prompt(sample)
            
            # Get model prediction
            response = await model.predict(prompt)
            
            if not response.success:
                return EvaluationResult(
                    model_name=model.get_name(),
                    sample_id=sample.poll_id,
                    prediction={},
                    ground_truth=sample.ground_truth,
                    success=False,
                    error=response.error,
                    response_time=time.time() - start_time,
                    raw_response=response.text
                )
            
            # Parse prediction
            prediction = self.parse_prediction(response.text, sample.input_data)
            
            return EvaluationResult(
                model_name=model.get_name(),
                sample_id=sample.poll_id,
                prediction=prediction,
                ground_truth=sample.ground_truth,
                success=True,
                response_time=time.time() - start_time,
                raw_response=response.text
            )
            
        except Exception as e:
            return EvaluationResult(
                model_name=model.get_name(),
                sample_id=sample.poll_id,
                prediction={},
                ground_truth=sample.ground_truth,
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def evaluate_model(self, model_name: str, samples: List[EvaluationSample]) -> List[EvaluationResult]:
        """Evaluate a model on all samples."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return []
        
        model = self.models[model_name]
        logger.info(f"Evaluating {model_name} on {len(samples)} samples")
        
        # Process in batches
        batch_size = self.config.batch_size
        results = []
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            
            # Create tasks for batch
            tasks = [self.evaluate_sample(model, sample) for sample in batch]
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch evaluation failed: {str(result)}")
                    results.append(EvaluationResult(
                        model_name=model_name,
                        sample_id="unknown",
                        prediction={},
                        ground_truth={},
                        success=False,
                        error=str(result)
                    ))
                else:
                    results.append(result)
            
            logger.info(f"Completed batch {i//batch_size + 1}/{(len(samples)-1)//batch_size + 1}")
        
        # Calculate success rate
        success_rate = sum(1 for r in results if r.success) / len(results)
        logger.info(f"Model {model_name} success rate: {success_rate:.2%}")
        
        return results
    
    async def evaluate_all_models(self, samples: List[EvaluationSample]) -> Dict[str, List[EvaluationResult]]:
        """Evaluate all loaded models."""
        all_results = {}
        
        for model_name in self.models:
            logger.info(f"Starting evaluation for {model_name}")
            results = await self.evaluate_model(model_name, samples)
            all_results[model_name] = results
            
            # Save intermediate results
            if self.config.save_intermediate:
                self.save_results({model_name: results}, 
                                f"{model_name}_intermediate_results.json")
        
        return all_results
    
    def compute_metrics(self, results: Dict[str, List[EvaluationResult]]) -> Dict[str, Dict[str, float]]:
        """Compute evaluation metrics for all models."""
        metrics = {}
        
        for model_name, model_results in results.items():
            # Filter successful predictions
            successful_results = [r for r in model_results if r.success]
            
            if not successful_results:
                logger.warning(f"No successful predictions for {model_name}")
                metrics[model_name] = {}
                continue
            
            # Extract predictions and ground truth
            predictions = [r.prediction for r in successful_results]
            ground_truth = [r.ground_truth for r in successful_results]
            
            # Compute metrics
            model_metrics = self.metrics.compute_all_metrics(predictions, ground_truth)
            metrics[model_name] = model_metrics
            
            logger.info(f"Computed {len(model_metrics)} metrics for {model_name}")
        
        return metrics
    
    def save_results(self, results: Dict[str, List[EvaluationResult]], filename: str):
        """Save evaluation results to file."""
        output_path = Path(self.config.output_dir) / filename
        
        # Convert results to serializable format
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = [
                {
                    'model_name': r.model_name,
                    'sample_id': r.sample_id,
                    'prediction': r.prediction,
                    'ground_truth': r.ground_truth,
                    'success': r.success,
                    'error': r.error,
                    'response_time': r.response_time,
                    'raw_response': r.raw_response
                }
                for r in model_results
            ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved results to {output_path}")
    
    def save_metrics(self, metrics: Dict[str, Dict[str, float]], filename: str):
        """Save evaluation metrics to file."""
        output_path = Path(self.config.output_dir) / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metrics to {output_path}")
    
    def generate_report(self, results: Dict[str, List[EvaluationResult]], 
                       metrics: Dict[str, Dict[str, float]]) -> str:
        """Generate evaluation report."""
        report = []
        report.append("# MindVote Poll Prediction Evaluation Report")
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        report.append("## Summary Statistics")
        for model_name, model_results in results.items():
            success_rate = sum(1 for r in model_results if r.success) / len(model_results)
            avg_response_time = sum(r.response_time for r in model_results) / len(model_results)
            
            report.append(f"- **{model_name}**:")
            report.append(f"  - Total samples: {len(model_results)}")
            report.append(f"  - Success rate: {success_rate:.2%}")
            report.append(f"  - Average response time: {avg_response_time:.2f}s")
            report.append("")
        
        # Metrics comparison
        report.append("## Metrics Comparison")
        if metrics:
            # Create metrics table
            metric_names = list(list(metrics.values())[0].keys())
            
            # Header
            report.append("| Model | " + " | ".join(metric_names) + " |")
            report.append("|" + "---|" * (len(metric_names) + 1))
            
            # Rows
            for model_name, model_metrics in metrics.items():
                row = [model_name]
                for metric_name in metric_names:
                    value = model_metrics.get(metric_name, 0)
                    row.append(f"{value:.3f}")
                report.append("| " + " | ".join(row) + " |")
        
        return "\n".join(report)
    
    async def run_evaluation(self, model_names: List[str] = None, 
                           max_samples: int = None) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        try:
            # Load models
            self.load_models(model_names)
            
            if not self.models:
                raise ValueError("No models loaded")
            
            # Prepare data
            samples = self.prepare_data()
            
            if max_samples:
                samples = samples[:max_samples]
            
            # Run evaluation
            results = await self.evaluate_all_models(samples)
            
            # Compute metrics
            metrics = self.compute_metrics(results)
            
            # Save results
            self.save_results(results, "evaluation_results.json")
            self.save_metrics(metrics, "evaluation_metrics.json")
            
            # Generate report
            report = self.generate_report(results, metrics)
            with open(Path(self.config.output_dir) / "evaluation_report.md", 'w') as f:
                f.write(report)
            
            logger.info("Evaluation completed successfully")
            
            return {
                'results': results,
                'metrics': metrics,
                'report': report
            }
            
        finally:
            # Cleanup models
            if self.models:
                ModelFactory.cleanup_models(self.models) 