#!/usr/bin/env python3
"""
Evaluation script for VLM Instruction Predictor

Usage:
    python eval_vlm_instruction_predictor.py --model_path exp/vlm_instr_predictor/best_model.pth --eval_split valid_seen
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.append(os.path.join(os.environ.get('ALFRED_ROOT', '.'), 'models'))

from models.model.vlm_instruction_predictor import Module
from models.utils.metric import compute_f1, compute_exact


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    args = checkpoint['args']
    vocab_dict = checkpoint['vocab']
    
    # Convert vocab to namespace-like object
    class VocabDict:
        def __init__(self, vocab_dict):
            self.vocab_dict = vocab_dict
            
        def __getitem__(self, key):
            return VocabNamespace(self.vocab_dict[key])
            
        def get(self, key, default=None):
            if key in self.vocab_dict:
                return VocabNamespace(self.vocab_dict[key])
            return default
    
    class VocabNamespace:
        def __init__(self, vocab_data):
            self.index2word = vocab_data['index2word']
            self.word2index_dict = vocab_data['word2index']
            
        def word2index(self, word, train=False):
            return self.word2index_dict.get(word, self.word2index_dict.get('<unk>', 3))
                
        def __len__(self):
            return len(self.index2word)
    
    vocab = VocabDict(vocab_dict)
    
    # Create model
    model = Module(args, vocab)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    return model, args, vocab


def load_data(data_path, split_name):
    """Load evaluation data"""
    splits_path = os.path.join(os.path.dirname(data_path), 'splits', 'oct21.json')
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    if split_name not in splits:
        raise ValueError(f"Split '{split_name}' not found in {splits_path}")
    
    data = splits[split_name]
    print(f"Loaded {len(data)} samples from {split_name} split")
    
    return data


def create_dummy_image():
    """Create a dummy image for testing"""
    return Image.new('RGB', (336, 336), color='black')


def evaluate_model(model, data, args, vocab, device, max_samples=None):
    """Evaluate the model on given data"""
    
    if max_samples:
        data = data[:max_samples]
    
    predictions = []
    ground_truths = []
    individual_metrics = []
    
    print("Evaluating model...")
    
    for i, sample in enumerate(tqdm(data)):
        try:
            # Load trajectory data
            if isinstance(sample, dict):
                traj_data = sample
            else:
                # Assume it's a path or similar
                traj_data = sample
            
            # Extract text data
            if 'turk_annotations' in traj_data and 'anns' in traj_data['turk_annotations']:
                ann = traj_data['turk_annotations']['anns'][0]
                goal_text = ann.get('task_desc', '')
                high_descs = ann.get('high_descs', [])
                
                if not high_descs:
                    continue
                
                # Use first instruction as context, predict the rest
                context_instr = high_descs[0] if len(high_descs) > 0 else ""
                target_instrs = high_descs[1:] if len(high_descs) > 1 else high_descs
                
                # Join target instructions
                target_text = " ".join(target_instrs)
                
            else:
                continue
            
            # Create dummy image (in real scenario, load actual image)
            image = create_dummy_image()
            
            # Generate prediction
            with torch.no_grad():
                predicted_words = model.vlm_model.generate_instruction(
                    goal_text, context_instr, image
                )
                predicted_text = " ".join(predicted_words)
            
            # Store results
            predictions.append(predicted_text)
            ground_truths.append(target_text)
            
            # Compute individual metrics
            f1_score = compute_f1(predicted_text.split(), target_text.split())
            exact_match = compute_exact(predicted_text, target_text)
            
            individual_metrics.append({
                'f1': f1_score,
                'exact_match': exact_match,
                'predicted': predicted_text,
                'target': target_text,
                'goal': goal_text,
                'context': context_instr
            })
            
            # Print progress
            if (i + 1) % 10 == 0:
                avg_f1 = np.mean([m['f1'] for m in individual_metrics])
                avg_em = np.mean([m['exact_match'] for m in individual_metrics])
                print(f"Progress: {i+1}/{len(data)}, F1: {avg_f1:.3f}, EM: {avg_em:.3f}")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    return predictions, ground_truths, individual_metrics


def compute_metrics(predictions, ground_truths, individual_metrics):
    """Compute overall metrics"""
    
    if not individual_metrics:
        return {}
    
    # Overall metrics
    f1_scores = [m['f1'] for m in individual_metrics]
    exact_matches = [m['exact_match'] for m in individual_metrics]
    
    metrics = {
        'num_samples': len(individual_metrics),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'exact_match_mean': np.mean(exact_matches),
        'exact_match_std': np.std(exact_matches),
        'f1_scores': f1_scores,
        'exact_matches': exact_matches
    }
    
    return metrics


def print_results(metrics, individual_metrics, num_examples=5):
    """Print evaluation results"""
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"F1 Score: {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"Exact Match: {metrics['exact_match_mean']:.4f} ± {metrics['exact_match_std']:.4f}")
    
    print(f"\nExample Predictions (showing first {num_examples}):")
    print("-" * 50)
    
    for i, item in enumerate(individual_metrics[:num_examples]):
        print(f"\nExample {i+1}:")
        print(f"Goal: {item['goal']}")
        print(f"Context: {item['context']}")
        print(f"Target: {item['target']}")
        print(f"Predicted: {item['predicted']}")
        print(f"F1: {item['f1']:.3f}, EM: {item['exact_match']}")
    
    # Show some statistics
    print(f"\nF1 Score Distribution:")
    print(f"  Min: {min(metrics['f1_scores']):.3f}")
    print(f"  Max: {max(metrics['f1_scores']):.3f}")
    print(f"  Median: {np.median(metrics['f1_scores']):.3f}")
    
    print(f"\nExact Match Distribution:")
    em_rate = np.mean(metrics['exact_matches'])
    print(f"  Exact Match Rate: {em_rate:.1%}")


def save_results(predictions, ground_truths, individual_metrics, metrics, output_path):
    """Save evaluation results to file"""
    
    results = {
        'predictions': predictions,
        'ground_truths': ground_truths,
        'individual_metrics': individual_metrics,
        'overall_metrics': {
            'num_samples': metrics['num_samples'],
            'f1_mean': float(metrics['f1_mean']),
            'f1_std': float(metrics['f1_std']),
            'exact_match_mean': float(metrics['exact_match_mean']),
            'exact_match_std': float(metrics['exact_match_std'])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLM Instruction Predictor')
    
    # Model and data
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='data/json_feat_2.1.0', help='Dataset directory')
    parser.add_argument('--eval_split', type=str, default='valid_seen', 
                       choices=['train', 'valid_seen', 'valid_unseen'], help='Evaluation split')
    
    # Evaluation settings
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to evaluate')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save results')
    parser.add_argument('--num_examples', type=int, default=10, help='Number of examples to print')
    
    # System
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, model_args, vocab = load_model(args.model_path, device)
    
    # Load data
    data = load_data(args.data, args.eval_split)
    
    # Evaluate
    predictions, ground_truths, individual_metrics = evaluate_model(
        model, data, model_args, vocab, device, args.max_samples
    )
    
    if not individual_metrics:
        print("No valid samples found for evaluation!")
        return
    
    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths, individual_metrics)
    
    # Print results
    print_results(metrics, individual_metrics, args.num_examples)
    
    # Save results
    if args.output_path:
        save_results(predictions, ground_truths, individual_metrics, metrics, args.output_path)
    else:
        # Auto-generate output path
        model_dir = os.path.dirname(args.model_path)
        output_path = os.path.join(model_dir, f'eval_results_{args.eval_split}.json')
        save_results(predictions, ground_truths, individual_metrics, metrics, output_path)


if __name__ == '__main__':
    main()
