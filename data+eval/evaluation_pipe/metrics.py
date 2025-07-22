"""
Evaluation metrics for poll prediction assessment.
"""
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy.stats import wasserstein_distance
import logging

logger = logging.getLogger(__name__)

class PollMetrics:
    """Calculate 1-Wasserstein distance for poll prediction evaluation."""
    
    def __init__(self):
        self.epsilon = 1e-8  # Small constant to avoid division by zero
    
    def compute_all_metrics(self, predictions: List[Dict[str, Any]], 
                          ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute 1-Wasserstein distance metric."""
        metrics = {}
        
        # Extract prediction and ground truth arrays
        pred_arrays, gt_arrays = self._extract_arrays(predictions, ground_truth)
        
        if len(pred_arrays) == 0:
            logger.warning("No valid predictions to evaluate")
            return metrics
        
        # 1-Wasserstein distance (Earth Mover's Distance)
        metrics['wasserstein_1'] = self.wasserstein_1_distance(pred_arrays, gt_arrays)
        
        return metrics
    
    def _extract_arrays(self, predictions: List[Dict[str, Any]], 
                       ground_truth: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract percentage arrays from predictions and ground truth."""
        pred_arrays = []
        gt_arrays = []
        
        for pred, gt in zip(predictions, ground_truth):
            try:
                pred_percentages = self._extract_percentages(pred)
                gt_percentages = self._extract_percentages(gt)
                
                if len(pred_percentages) == len(gt_percentages) and len(pred_percentages) > 0:
                    pred_arrays.append(np.array(pred_percentages))
                    gt_arrays.append(np.array(gt_percentages))
                else:
                    logger.warning(f"Mismatched option counts: pred={len(pred_percentages)}, gt={len(gt_percentages)}")
            except Exception as e:
                logger.warning(f"Error extracting percentages: {str(e)}")
                continue
        
        return pred_arrays, gt_arrays
    
    def _extract_percentages(self, data: Dict[str, Any]) -> List[float]:
        """Extract percentage values from poll data."""
        percentages = []
        
        if 'options' in data:
            for option in data['options']:
                if 'percentage' in option:
                    pct = option['percentage']
                    if isinstance(pct, (int, float)):
                        percentages.append(float(pct))
                    elif isinstance(pct, str) and pct.lower() != 'guess here':
                        try:
                            percentages.append(float(pct))
                        except ValueError:
                            logger.warning(f"Could not parse percentage: {pct}")
        
        return percentages
    
    def wasserstein_1_distance(self, predictions: List[np.ndarray], 
                              ground_truth: List[np.ndarray]) -> float:
        """Calculate 1-Wasserstein distance (Earth Mover's Distance)."""
        distances = []
        
        for pred, gt in zip(predictions, ground_truth):
            # Normalize to probability distributions
            pred_norm = pred / (np.sum(pred) + self.epsilon)
            gt_norm = gt / (np.sum(gt) + self.epsilon)
            
            # Create position arrays (option indices)
            positions = np.arange(len(pred))
            
            # Calculate 1-Wasserstein distance
            distance = wasserstein_distance(positions, positions, pred_norm, gt_norm)
            distances.append(distance)
        
        return np.mean(distances) if distances else 0.0

def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute aggregate metrics across all models."""
    metrics = PollMetrics()
    aggregate_results = {}
    
    for result in results:
        model_name = result['model_name']
        predictions = result['predictions']
        ground_truth = result['ground_truth']
        
        model_metrics = metrics.compute_all_metrics(predictions, ground_truth)
        aggregate_results[model_name] = model_metrics
    
    return aggregate_results 