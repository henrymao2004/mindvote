"""
Data processing utilities for the Mindvote evaluation pipeline.
"""
import json
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class PollData:
    """Structure for individual poll data."""
    question: str
    total_votes: int
    options: List[Dict[str, Any]]
    context: str = ""
    metadata: Dict[str, Any] = None
    platform: str = ""
    domain: str = ""
    date: str = ""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class EvaluationSample:
    """Structure for evaluation sample with ground truth and test input."""
    poll_id: str
    question: str
    context: str
    input_data: Dict[str, Any]
    ground_truth: Dict[str, Any]
    holdout_options: List[str]
    metadata: Dict[str, Any]

class DataProcessor:
    """Handles loading and preprocessing of Mindvote data."""
    
    def __init__(self, data_dir: str = "data", random_seed: int = 42):
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def load_poll_data(self) -> List[PollData]:
        """Load poll data from JSON files."""
        polls = []
        
        # Load metadata
        metadata_path = self.data_dir / "poll_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
                for item in metadata_list:
                    question = item.get('question', '')
                    metadata[question] = item
        
        # Load Reddit polls
        reddit_path = self.data_dir / "reddit-pollpart.json"
        if reddit_path.exists():
            with open(reddit_path, 'r', encoding='utf-8') as f:
                reddit_data = json.load(f)
                for item in reddit_data:
                    question = item.get('question', '')
                    meta = metadata.get(question, {})
                    
                    context = self._build_context(meta)
                    
                    poll = PollData(
                        question=question,
                        total_votes=item.get('totalVotes', 0),
                        options=item.get('options', []),
                        context=context,
                        metadata=meta,
                        platform="Reddit",
                        domain=meta.get('domain', ''),
                        date=meta.get('date', '')
                    )
                    polls.append(poll)
        
        # Load Weibo polls
        weibo_path = self.data_dir / "weibo_pollpart.json"
        if weibo_path.exists():
            with open(weibo_path, 'r', encoding='utf-8') as f:
                weibo_data = json.load(f)
                for item in weibo_data:
                    question = item.get('question', '')
                    
                    # For Weibo, use the context field directly
                    context = item.get('context', '')
                    
                    poll = PollData(
                        question=question,
                        total_votes=item.get('totalVotes', 0),
                        options=item.get('options', []),
                        context=context,
                        metadata={},
                        platform="Weibo",
                        domain=item.get('topic', ''),
                        date=""
                    )
                    polls.append(poll)
        
        logger.info(f"Loaded {len(polls)} polls from data directory")
        return polls
    
    def _build_context(self, metadata: Dict[str, Any]) -> str:
        """Build context string from metadata."""
        context_parts = []
        
        if metadata.get('platformContext'):
            context_parts.append(f"Platform Context: {metadata['platformContext']}")
        
        if metadata.get('domainContext'):
            context_parts.append(f"Domain Context: {metadata['domainContext']}")
        
        if metadata.get('temporalContext'):
            context_parts.append(f"Temporal Context: {metadata['temporalContext']}")
        
        return "\n\n".join(context_parts)
    
    def filter_polls(self, polls: List[PollData], max_options: int = 10) -> List[PollData]:
        """Filter polls based on criteria."""
        filtered = []
        
        for poll in polls:
            # Filter by maximum options if specified
            if max_options > 0 and len(poll.options) > max_options:
                continue
            
            # Filter polls with valid percentages
            valid_options = []
            for option in poll.options:
                if 'percentage' in option and isinstance(option['percentage'], (int, float)):
                    valid_options.append(option)
            
            if len(valid_options) < 2:
                continue
            
            poll.options = valid_options
            filtered.append(poll)
        
        logger.info(f"Filtered to {len(filtered)} polls after applying criteria")
        return filtered
    
    def create_evaluation_samples(self, polls: List[PollData], holdout_percentage: float = 0.3) -> List[EvaluationSample]:
        """Create evaluation samples with held-out options."""
        samples = []
        
        for i, poll in enumerate(polls):
            # Determine how many options to hold out
            n_options = len(poll.options)
            if n_options < 2:
                continue
            
            n_holdout = max(1, int(n_options * holdout_percentage))
            n_holdout = min(n_holdout, n_options - 1)  # Always keep at least one option
            
            # Randomly select options to hold out
            holdout_indices = random.sample(range(n_options), n_holdout)
            holdout_options = [poll.options[idx]['optionText'] for idx in holdout_indices]
            
            # Create input data with "Guess Here" for holdout options
            input_data = {
                "question": poll.question,
                "totalVotes": poll.total_votes,
                "date": poll.date,
                "options": []
            }
            
            ground_truth = {
                "question": poll.question,
                "totalVotes": poll.total_votes,
                "date": poll.date,
                "options": []
            }
            
            for idx, option in enumerate(poll.options):
                input_option = {
                    "optionText": option['optionText'],
                    "percentage": "Guess Here" if idx in holdout_indices else option['percentage']
                }
                input_data["options"].append(input_option)
                
                ground_truth_option = {
                    "optionText": option['optionText'],
                    "percentage": option['percentage']
                }
                ground_truth["options"].append(ground_truth_option)
            
            sample = EvaluationSample(
                poll_id=f"poll_{i}",
                question=poll.question,
                context=poll.context,
                input_data=input_data,
                ground_truth=ground_truth,
                holdout_options=holdout_options,
                metadata={
                    "platform": poll.platform,
                    "domain": poll.domain,
                    "date": poll.date,
                    "total_votes": poll.total_votes,
                    "n_options": n_options,
                    "n_holdout": n_holdout
                }
            )
            samples.append(sample)
        
        logger.info(f"Created {len(samples)} evaluation samples")
        return samples
    
    def split_data(self, samples: List[EvaluationSample], test_split: float = 0.2) -> Tuple[List[EvaluationSample], List[EvaluationSample]]:
        """Split data into train and test sets."""
        random.shuffle(samples)
        split_idx = int(len(samples) * (1 - test_split))
        
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]
        
        logger.info(f"Split data: {len(train_samples)} train, {len(test_samples)} test")
        return train_samples, test_samples
    
    def save_samples(self, samples: List[EvaluationSample], output_path: str):
        """Save evaluation samples to JSON file."""
        output_data = []
        for sample in samples:
            output_data.append({
                "poll_id": sample.poll_id,
                "question": sample.question,
                "context": sample.context,
                "input_data": sample.input_data,
                "ground_truth": sample.ground_truth,
                "holdout_options": sample.holdout_options,
                "metadata": sample.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(samples)} samples to {output_path}")
    
    def load_samples(self, input_path: str) -> List[EvaluationSample]:
        """Load evaluation samples from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = EvaluationSample(
                poll_id=item['poll_id'],
                question=item['question'],
                context=item['context'],
                input_data=item['input_data'],
                ground_truth=item['ground_truth'],
                holdout_options=item['holdout_options'],
                metadata=item['metadata']
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from {input_path}")
        return samples

def get_data_statistics(samples: List[EvaluationSample]) -> Dict[str, Any]:
    """Compute statistics about the evaluation data."""
    stats = {
        "total_samples": len(samples),
        "platforms": {},
        "domains": {},
        "options_per_poll": [],
        "holdout_per_poll": [],
        "total_votes": []
    }
    
    for sample in samples:
        # Platform distribution
        platform = sample.metadata.get('platform', 'Unknown')
        stats["platforms"][platform] = stats["platforms"].get(platform, 0) + 1
        
        # Domain distribution
        domain = sample.metadata.get('domain', 'Unknown')
        stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
        
        # Options and holdout statistics
        stats["options_per_poll"].append(sample.metadata.get('n_options', 0))
        stats["holdout_per_poll"].append(sample.metadata.get('n_holdout', 0))
        stats["total_votes"].append(sample.metadata.get('total_votes', 0))
    
    # Convert to numpy arrays for statistics
    stats["options_per_poll"] = np.array(stats["options_per_poll"])
    stats["holdout_per_poll"] = np.array(stats["holdout_per_poll"])
    stats["total_votes"] = np.array(stats["total_votes"])
    
    # Add summary statistics
    stats["avg_options"] = np.mean(stats["options_per_poll"])
    stats["avg_holdout"] = np.mean(stats["holdout_per_poll"])
    stats["avg_votes"] = np.mean(stats["total_votes"])
    
    return stats 