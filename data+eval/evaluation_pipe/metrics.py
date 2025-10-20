"""
Evaluation metrics for poll prediction assessment.
Following OpinionQA benchmark methodology for option handling.
"""
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy.stats import wasserstein_distance, entropy, spearmanr
import logging

logger = logging.getLogger(__name__)

class PollMetrics:
    """Calculate evaluation metrics for poll prediction assessment."""
    
    def __init__(self):
        self.epsilon = 1e-8  # Small constant to avoid division by zero
        # Hedging option patterns (case-insensitive)
        self.hedging_patterns = [
            'neither', 'about the same', 'i don\'t know', 'other', 
            'unsure', 'not sure', 'no opinion', 'neutral'
        ]
        # Refused option patterns (case-insensitive) - in poll setting these are non-ordinal options
        self.refused_patterns = ['results', 'other', 'other options']
    
    def compute_all_metrics(self, predictions: List[Dict[str, Any]], 
                          ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute all evaluation metrics with OpinionQA option handling."""
        metrics = {}
        
        # Extract prediction and ground truth arrays with special option handling
        pred_arrays, gt_arrays, refused_stats = self._extract_arrays_with_filtering(predictions, ground_truth)
        
        if len(pred_arrays) == 0:
            logger.warning("No valid predictions to evaluate")
            return metrics
        
        # 1-Wasserstein distance (Earth Mover's Distance) on ordinal options only
        metrics['wasserstein_1'] = self.wasserstein_1_distance(pred_arrays, gt_arrays)
        
        # KL divergence (Kullback-Leibler divergence)
        metrics['kl_divergence'] = self.kl_divergence(pred_arrays, gt_arrays)
        
        # Spearman rank correlation coefficient
        metrics['spearman_correlation'] = self.spearman_rank_correlation(pred_arrays, gt_arrays)
        
        # One-hot accuracy (top-1 accuracy)
        metrics['one_hot_accuracy'] = self.one_hot_accuracy(pred_arrays, gt_arrays)
        
        # Add non-ordinal option comparison if present
        if refused_stats['count'] > 0:
            metrics['non_ordinal_accuracy'] = refused_stats['accuracy']
            metrics['non_ordinal_samples'] = refused_stats['count']
        
        return metrics
    
    def _extract_arrays_with_filtering(self, predictions: List[Dict[str, Any]], 
                                     ground_truth: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
        """Extract percentage arrays with OpinionQA option filtering."""
        pred_arrays = []
        gt_arrays = []
        refused_stats = {'count': 0, 'correct': 0, 'accuracy': 0.0}
        
        for pred, gt in zip(predictions, ground_truth):
            try:
                # Process with special option handling
                pred_processed = self._process_options_with_filtering(pred)
                gt_processed = self._process_options_with_filtering(gt)
                
                # Extract ordinal percentages
                pred_percentages = pred_processed['ordinal_percentages']
                gt_percentages = gt_processed['ordinal_percentages']
                
                if len(pred_percentages) == len(gt_percentages) and len(pred_percentages) > 0:
                    pred_arrays.append(np.array(pred_percentages))
                    gt_arrays.append(np.array(gt_percentages))
                    
                    # Track non-ordinal option accuracy
                    if pred_processed['has_refused'] or gt_processed['has_refused']:
                        refused_stats['count'] += 1
                        if abs(pred_processed['refused_pct'] - gt_processed['refused_pct']) < 0.01:
                            refused_stats['correct'] += 1
                else:
                    logger.warning(f"Mismatched ordinal option counts: pred={len(pred_percentages)}, gt={len(gt_percentages)}")
            except Exception as e:
                logger.warning(f"Error processing options: {str(e)}")
                continue
        
        # Calculate non-ordinal option accuracy
        if refused_stats['count'] > 0:
            refused_stats['accuracy'] = refused_stats['correct'] / refused_stats['count']
        
        return pred_arrays, gt_arrays, refused_stats
    
    def _process_options_with_filtering(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process options following OpinionQA methodology."""
        result = {
            'ordinal_percentages': [],
            'has_refused': False,
            'refused_pct': 0.0,
            'hedging_indices': []
        }
        
        if 'options' not in data:
            return result
        
        options = data['options']
        ordinal_percentages = []
        refused_pct = 0.0
        hedging_percentages = []
        hedging_indices = []
        
        for i, option in enumerate(options):
            if 'percentage' not in option:
                continue
                
            pct = self._extract_single_percentage(option['percentage'])
            if pct is None:
                continue
            
            option_text = option.get('text', '').lower().strip()
            
            # Check if this is a non-ordinal option (results/other)
            if self._is_refused_option(option_text):
                result['has_refused'] = True
                refused_pct += pct
                continue
            
            # Check if this is a hedging option
            if self._is_hedging_option(option_text):
                hedging_percentages.append(pct)
                hedging_indices.append(i)
                continue
            
            # Regular ordinal option
            ordinal_percentages.append(pct)
        
        # Handle hedging options: map to mean of ordinal options
        if hedging_percentages and ordinal_percentages:
            mean_ordinal = np.mean(ordinal_percentages)
            for hedging_pct in hedging_percentages:
                ordinal_percentages.append(mean_ordinal)
            result['hedging_indices'] = hedging_indices
        
        result['ordinal_percentages'] = ordinal_percentages
        result['refused_pct'] = refused_pct
        
        return result
    
    def _extract_single_percentage(self, pct_value: Any) -> float:
        """Extract a single percentage value."""
        if isinstance(pct_value, (int, float)):
            return float(pct_value)
        elif isinstance(pct_value, str) and pct_value.lower() != 'guess here':
            try:
                return float(pct_value)
            except ValueError:
                logger.warning(f"Could not parse percentage: {pct_value}")
                return None
        return None
    
    def _is_refused_option(self, option_text: str) -> bool:
        """Check if option represents a non-ordinal option (results/other)."""
        return any(pattern in option_text for pattern in self.refused_patterns)
    
    def _is_hedging_option(self, option_text: str) -> bool:
        """Check if option represents hedging (non-ordinal neutral response)."""
        return any(pattern in option_text for pattern in self.hedging_patterns)
    
    def kl_divergence(self, predictions: List[np.ndarray], 
                     ground_truth: List[np.ndarray]) -> float:
        """Calculate KL divergence between predicted and ground truth distributions."""
        divergences = []
        
        for pred, gt in zip(predictions, ground_truth):
            # Normalize to probability distributions
            pred_norm = pred / (np.sum(pred) + self.epsilon)
            gt_norm = gt / (np.sum(gt) + self.epsilon)
            
            # Add small epsilon to avoid log(0)
            pred_norm = pred_norm + self.epsilon
            gt_norm = gt_norm + self.epsilon
            
            # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
            # Here P is ground truth, Q is prediction
            kl_div = entropy(gt_norm, pred_norm)
            divergences.append(kl_div)
        
        return np.mean(divergences) if divergences else 0.0
    
    def spearman_rank_correlation(self, predictions: List[np.ndarray], 
                                ground_truth: List[np.ndarray]) -> float:
        """Calculate Spearman rank correlation between predicted and ground truth rankings."""
        correlations = []
        
        for pred, gt in zip(predictions, ground_truth):
            # Normalize to probability distributions
            pred_norm = pred / (np.sum(pred) + self.epsilon)
            gt_norm = gt / (np.sum(gt) + self.epsilon)
            
            # Calculate Spearman rank correlation
            try:
                correlation, _ = spearmanr(pred_norm, gt_norm)
                if not np.isnan(correlation):
                    correlations.append(correlation)
            except Exception as e:
                logger.warning(f"Error calculating Spearman correlation: {str(e)}")
                continue
        
        return np.mean(correlations) if correlations else 0.0
    
    def one_hot_accuracy(self, predictions: List[np.ndarray], 
                        ground_truth: List[np.ndarray]) -> float:
        """Calculate one-hot (top-1) accuracy between predicted and ground truth."""
        correct_predictions = 0
        total_predictions = 0
        
        for pred, gt in zip(predictions, ground_truth):
            # Normalize to probability distributions
            pred_norm = pred / (np.sum(pred) + self.epsilon)
            gt_norm = gt / (np.sum(gt) + self.epsilon)
            
            # Find the index of maximum probability for both
            pred_top_choice = np.argmax(pred_norm)
            gt_top_choice = np.argmax(gt_norm)
            
            if pred_top_choice == gt_top_choice:
                correct_predictions += 1
            total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0

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