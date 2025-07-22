"""
Utility functions for the evaluation pipeline.
"""
import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def save_json(data: Dict[str, Any], filepath: str):
    """Save data as JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_metrics_comparison_plot(metrics: Dict[str, Dict[str, float]], 
                                 output_path: str = "wasserstein_comparison.png"):
    """Create visualization comparing 1-Wasserstein distance across models."""
    if not metrics:
        logger.warning("No metrics to plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics).T
    
    # Create single plot for 1-Wasserstein distance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'wasserstein_1' in df.columns:
        df['wasserstein_1'].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('1-Wasserstein Distance Comparison (Lower is Better)', fontsize=14)
        ax.set_ylabel('1-Wasserstein Distance')
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(df['wasserstein_1']):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved 1-Wasserstein distance comparison plot to {output_path}")

def create_model_ranking_plot(metrics: Dict[str, Dict[str, float]], 
                             output_path: str = "model_ranking.png"):
    """Create model ranking visualization based on 1-Wasserstein distance."""
    if not metrics:
        logger.warning("No metrics to plot")
        return
    
    if 'wasserstein_1' not in list(metrics.values())[0]:
        logger.warning("1-Wasserstein metric not available for ranking")
        return
    
    # Extract 1-Wasserstein distances and sort (lower is better)
    model_scores = {}
    for model_name, model_metrics in metrics.items():
        # For Wasserstein distance, lower is better, so we negate for ranking
        model_scores[model_name] = -model_metrics.get('wasserstein_1', 0)
    
    # Sort models by score (higher negated score = lower distance = better)
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create plot with actual Wasserstein distances (not negated)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models, _ = zip(*sorted_models)
    distances = [metrics[model]['wasserstein_1'] for model in models]
    bars = ax.bar(models, distances)
    
    # Color bars from green (best) to red (worst)
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_title('Model Ranking by 1-Wasserstein Distance (Lower is Better)')
    ax.set_ylabel('1-Wasserstein Distance')
    ax.set_xlabel('Model')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, distance in zip(bars, distances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{distance:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved model ranking plot to {output_path}")

def create_success_rate_plot(results: Dict[str, List[Dict[str, Any]]], 
                           output_path: str = "success_rates.png"):
    """Create success rate visualization."""
    success_rates = {}
    
    for model_name, model_results in results.items():
        if model_results:
            success_rate = sum(1 for r in model_results if r.get('success', False)) / len(model_results)
            success_rates[model_name] = success_rate
    
    if not success_rates:
        logger.warning("No success rates to plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(success_rates.keys())
    rates = list(success_rates.values())
    
    bars = ax.bar(models, rates)
    
    # Color bars based on success rate
    colors = ['red' if r < 0.5 else 'orange' if r < 0.8 else 'green' for r in rates]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_title('Model Success Rates')
    ax.set_ylabel('Success Rate')
    ax.set_xlabel('Model')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved success rate plot to {output_path}")

def create_response_time_plot(results: Dict[str, List[Dict[str, Any]]], 
                            output_path: str = "response_times.png"):
    """Create response time visualization."""
    response_times = {}
    
    for model_name, model_results in results.items():
        if model_results:
            times = [r.get('response_time', 0) for r in model_results if r.get('success', False)]
            if times:
                response_times[model_name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'median': np.median(times)
                }
    
    if not response_times:
        logger.warning("No response times to plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(response_times.keys())
    means = [response_times[m]['mean'] for m in models]
    stds = [response_times[m]['std'] for m in models]
    
    bars = ax.bar(models, means, yerr=stds, capsize=5)
    
    ax.set_title('Model Response Times')
    ax.set_ylabel('Response Time (seconds)')
    ax.set_xlabel('Model')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved response time plot to {output_path}")

def generate_summary_statistics(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Generate summary statistics from evaluation results."""
    summary = {}
    
    for model_name, model_results in results.items():
        if not model_results:
            continue
        
        successful_results = [r for r in model_results if r.get('success', False)]
        response_times = [r.get('response_time', 0) for r in successful_results]
        
        summary[model_name] = {
            'total_samples': len(model_results),
            'successful_predictions': len(successful_results),
            'success_rate': len(successful_results) / len(model_results) if model_results else 0,
            'average_response_time': np.mean(response_times) if response_times else 0,
            'median_response_time': np.median(response_times) if response_times else 0,
            'std_response_time': np.std(response_times) if response_times else 0,
            'min_response_time': np.min(response_times) if response_times else 0,
            'max_response_time': np.max(response_times) if response_times else 0
        }
    
    return summary

def validate_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate prediction format and completeness."""
    validation_results = {
        'total_predictions': len(predictions),
        'valid_predictions': 0,
        'invalid_predictions': 0,
        'sum_errors': 0,
        'format_errors': 0,
        'missing_fields': 0
    }
    
    for pred in predictions:
        is_valid = True
        
        # Check required fields
        if 'options' not in pred:
            validation_results['missing_fields'] += 1
            is_valid = False
            continue
        
        # Check option format
        percentages = []
        for option in pred['options']:
            if 'percentage' not in option:
                validation_results['format_errors'] += 1
                is_valid = False
                break
            
            try:
                pct = float(option['percentage'])
                percentages.append(pct)
            except (ValueError, TypeError):
                validation_results['format_errors'] += 1
                is_valid = False
                break
        
        if not is_valid:
            validation_results['invalid_predictions'] += 1
            continue
        
        # Check if percentages sum to ~100
        total = sum(percentages)
        if abs(total - 100) > 0.1:
            validation_results['sum_errors'] += 1
            is_valid = False
        
        if is_valid:
            validation_results['valid_predictions'] += 1
        else:
            validation_results['invalid_predictions'] += 1
    
    return validation_results

def create_evaluation_dashboard(results: Dict[str, List[Dict[str, Any]]], 
                              metrics: Dict[str, Dict[str, float]], 
                              output_dir: str = "dashboard"):
    """Create a comprehensive evaluation dashboard."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    create_metrics_comparison_plot(metrics, str(output_path / "wasserstein_comparison.png"))
    create_model_ranking_plot(metrics, str(output_path / "model_ranking.png"))
    create_success_rate_plot(results, str(output_path / "success_rates.png"))
    create_response_time_plot(results, str(output_path / "response_times.png"))
    
    # Generate summary statistics
    summary = generate_summary_statistics(results)
    save_json(summary, str(output_path / "summary_statistics.json"))
    
    # Create HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mindvote Evaluation Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            .plot img {{ max-width: 100%; height: auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Mindvote Poll Prediction Evaluation Dashboard</h1>
            <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
                         <div class="plot">
                 <h2>1-Wasserstein Distance Comparison</h2>
                 <img src="wasserstein_comparison.png" alt="1-Wasserstein Distance Comparison">
             </div>
            
            <div class="plot">
                <h2>Model Ranking</h2>
                <img src="model_ranking.png" alt="Model Ranking">
            </div>
            
            <div class="plot">
                <h2>Success Rates</h2>
                <img src="success_rates.png" alt="Success Rates">
            </div>
            
            <div class="plot">
                <h2>Response Times</h2>
                <img src="response_times.png" alt="Response Times">
            </div>
            
            <div class="metrics">
                <h2>Detailed Metrics</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Success Rate</th>
                        <th>1-Wasserstein Distance</th>
                        <th>Avg Response Time</th>
                    </tr>
    """
    
    for model_name in results.keys():
        model_summary = summary.get(model_name, {})
        model_metrics = metrics.get(model_name, {})
        
        html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{model_summary.get('success_rate', 0):.2%}</td>
                        <td>{model_metrics.get('wasserstein_1', 0):.3f}</td>
                        <td>{model_summary.get('average_response_time', 0):.1f}s</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path / "dashboard.html", 'w') as f:
        f.write(html_content)
    
    logger.info(f"Created evaluation dashboard at {output_path}")

def export_results_to_csv(results: Dict[str, List[Dict[str, Any]]], 
                         output_path: str = "results.csv"):
    """Export evaluation results to CSV format."""
    rows = []
    
    for model_name, model_results in results.items():
        for result in model_results:
            row = {
                'model_name': model_name,
                'sample_id': result.get('sample_id', ''),
                'success': result.get('success', False),
                'response_time': result.get('response_time', 0),
                'error': result.get('error', '')
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Exported results to {output_path}")

def compare_models_statistical(metrics: Dict[str, Dict[str, float]], 
                             metric_name: str = 'wasserstein_1') -> Dict[str, Any]:
    """Perform statistical comparison between models based on 1-Wasserstein distance."""
    if metric_name not in list(metrics.values())[0]:
        logger.warning(f"Metric {metric_name} not found")
        return {}
    
    model_names = list(metrics.keys())
    values = [metrics[model][metric_name] for model in model_names]
    
    # For Wasserstein distance, lower is better
    best_idx = np.argmin(values)
    worst_idx = np.argmax(values)
    
    return {
        'metric': metric_name,
        'best_model': model_names[best_idx],
        'best_value': values[best_idx],
        'worst_model': model_names[worst_idx],
        'worst_value': values[worst_idx],
        'mean_value': np.mean(values),
        'std_value': np.std(values),
        'all_values': dict(zip(model_names, values))
    } 