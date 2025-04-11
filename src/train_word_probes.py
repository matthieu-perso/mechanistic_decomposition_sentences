#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train Word Probes Script
------------------------
This script trains and evaluates various probes on word-level embeddings
to analyze linguistic properties encoded in transformer models.
"""

import os
import json
import pickle
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel


# === Model Classes ===

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class NonlinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class AdaptiveSoftmaxProbe(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        cutoffs = [1000, min(10000, n_classes - 2)] if n_classes > 10000 else [1000]
        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=input_dim,
            n_classes=n_classes,
            cutoffs=cutoffs,
            div_value=4.0
        )

    def forward(self, x, target=None):
        if target is not None:
            return self.adaptive_softmax(x, target)
        else:
            return self.adaptive_softmax.log_prob(x)


class NonlinearAdaptiveSoftmaxProbe(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        cutoffs = [1000, min(10000, n_classes - 2)] if n_classes > 10000 else [1000]
        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=hidden_dim,
            n_classes=n_classes,
            cutoffs=cutoffs,
            div_value=4.0
        )

    def forward(self, x, target=None):
        x = self.hidden(x)
        if target is not None:
            return self.adaptive_softmax(x, target)
        else:
            return self.adaptive_softmax.log_prob(x)


class RandomPredictionProbe(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        return torch.randint(0, self.n_classes, (x.size(0),), device=x.device)


# === Training Functions ===

def save_model(model, name, folder):
    """Save a trained model."""
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(folder, f"{name}.pt"))
    # Also save the full model for easier loading
    with open(os.path.join(folder, f"{name}.pkl"), 'wb') as f:
        pickle.dump(model, f)


def train_probe(model, X, y, task_name="TASK", epochs=10, output_dir="trained_models"):
    """Train a classification probe."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in tqdm(loader, desc=f"{task_name} Epoch {epoch+1}"):
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{task_name} - Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    save_model(model, task_name, output_dir)
    return model


def train_adaptive_probe(X, y, num_classes, nonlinear=False, hidden_dim=128, task_name="TASK", epochs=10, output_dir="trained_models"):
    """Train an adaptive softmax probe for large vocabulary tasks."""
    if nonlinear:
        model = NonlinearAdaptiveSoftmaxProbe(X.shape[1], num_classes, hidden_dim=hidden_dim)
    else:
        model = AdaptiveSoftmaxProbe(X.shape[1], num_classes)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in tqdm(loader, desc=f"{task_name} Epoch {epoch+1}"):
            optimizer.zero_grad()
            output = model(xb, yb)
            loss = output.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{task_name} - Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    save_model(model, task_name, output_dir)
    return model


# === Evaluation Functions ===

def evaluate_probe(model, X, y):
    """Evaluate a classification probe."""
    model.eval()
    with torch.no_grad():
        if isinstance(model, RandomPredictionProbe):
            preds = model(X)
        else:
            preds = model(X).argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
    return accuracy


def evaluate_adaptive_probe(model, X, y):
    """Evaluate an adaptive softmax probe."""
    model.eval()
    with torch.no_grad():
        log_probs = model(X).cpu()
        preds = torch.argmax(log_probs, dim=1)
        accuracy = (preds == y.cpu()).float().mean().item()
    return accuracy


# === Utility Functions ===

def save_label_encoders(le_pos, le_dep, le_word, folder):
    """Save label encoders for later use."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "le_pos.pkl"), 'wb') as f:
        pickle.dump(le_pos, f)
    with open(os.path.join(folder, "le_dep.pkl"), 'wb') as f:
        pickle.dump(le_dep, f)
    with open(os.path.join(folder, "le_word.pkl"), 'wb') as f:
        pickle.dump(le_word, f)


def save_word_representations(model, X, y_word, le_word, folder):
    """Extract and save word representations from the model."""
    model.eval()
    with torch.no_grad():
        reps = model.hidden(X).cpu()
    
    # Create a mapping from index to word
    idx_to_word = {i: w for w, i in zip(le_word.classes_, range(len(le_word.classes_)))}
    
    # Create a dictionary mapping words to their representations
    word_to_repr = {}
    for y, rep in zip(y_word, reps):
        word = idx_to_word[y.item()]
        if word not in word_to_repr:
            word_to_repr[word] = rep.numpy().tolist()
    
    # Save to JSON
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "word_to_repr.json"), "w") as f:
        json.dump(word_to_repr, f, indent=2)


def run_all_probes_and_controls(X, y_pos, y_dep, y_position, y_word, le_pos, le_dep, le_word, output_dir, hidden_dim=128):
    """Run all probes and control experiments."""
    results = {}

    save_label_encoders(le_pos, le_dep, le_word, output_dir)

    # === Main Probes ===
    print("\n=== Training Main Probes ===")
    pos_model = train_probe(LinearProbe(X.shape[1], len(le_pos.classes_)), X, y_pos, "POS", output_dir=output_dir)
    dep_model = train_probe(LinearProbe(X.shape[1], len(le_dep.classes_)), X, y_dep, "DEP", output_dir=output_dir)
    position_model = train_probe(LinearProbe(X.shape[1], y_position.max().item() + 1), X, y_position, "POSITION", output_dir=output_dir)
    word_model = train_adaptive_probe(X, y_word, len(le_word.classes_), task_name="WORD", output_dir=output_dir)

    # === Nonlinear Probes ===
    print("\n=== Training Nonlinear Probes ===")
    pos_nonlinear = train_probe(NonlinearProbe(X.shape[1], len(le_pos.classes_), hidden_dim=hidden_dim), X, y_pos, "POS_Nonlinear", output_dir=output_dir)
    dep_nonlinear = train_probe(NonlinearProbe(X.shape[1], len(le_dep.classes_), hidden_dim=hidden_dim), X, y_dep, "DEP_Nonlinear", output_dir=output_dir)
    position_nonlinear = train_probe(NonlinearProbe(X.shape[1], y_position.max().item() + 1, hidden_dim=hidden_dim), X, y_position, "POSITION_Nonlinear", output_dir=output_dir)
    word_nonlinear = train_adaptive_probe(X, y_word, len(le_word.classes_), nonlinear=True, hidden_dim=hidden_dim, task_name="WORD_Nonlinear", output_dir=output_dir)

    save_word_representations(word_nonlinear, X, y_word, le_word, output_dir)

    # === Random Baselines ===
    print("\n=== Evaluating Random Baselines ===")
    pos_random = RandomPredictionProbe(len(le_pos.classes_))
    dep_random = RandomPredictionProbe(len(le_dep.classes_))
    position_random = RandomPredictionProbe(y_position.max().item() + 1)
    word_random = RandomPredictionProbe(len(le_word.classes_))

    # === Shuffled Labels ===
    print("\n=== Training with Shuffled Labels ===")
    pos_shuffled = train_probe(LinearProbe(X.shape[1], len(le_pos.classes_)), X, y_pos[torch.randperm(len(y_pos))], "POS_Shuffled", output_dir=output_dir)
    dep_shuffled = train_probe(LinearProbe(X.shape[1], len(le_dep.classes_)), X, y_dep[torch.randperm(len(y_dep))], "DEP_Shuffled", output_dir=output_dir)
    position_shuffled = train_probe(LinearProbe(X.shape[1], y_position.max().item() + 1), X, y_position[torch.randperm(len(y_position))], "POSITION_Shuffled", output_dir=output_dir)
    word_shuffled = train_adaptive_probe(X, y_word[torch.randperm(len(y_word))], len(le_word.classes_), task_name="WORD_Shuffled", output_dir=output_dir)

    # === Random Representations ===
    print("\n=== Training with Random Representations ===")
    X_rand = torch.randn_like(X)
    pos_randrep = train_probe(LinearProbe(X.shape[1], len(le_pos.classes_)), X_rand, y_pos, "POS_RandomRep", output_dir=output_dir)
    dep_randrep = train_probe(LinearProbe(X.shape[1], len(le_dep.classes_)), X_rand, y_dep, "DEP_RandomRep", output_dir=output_dir)
    position_randrep = train_probe(LinearProbe(X.shape[1], y_position.max().item() + 1), X_rand, y_position, "POSITION_RandomRep", output_dir=output_dir)
    word_randrep = train_adaptive_probe(X_rand, y_word, len(le_word.classes_), task_name="WORD_RandomRep", output_dir=output_dir)

    # === Evaluation ===
    print("\n=== Evaluation ===")
    results = {
        "POS": {
            "linear": evaluate_probe(pos_model, X, y_pos),
            "nonlinear": evaluate_probe(pos_nonlinear, X, y_pos),
            "random": evaluate_probe(pos_random, X, y_pos),
            "shuffled": evaluate_probe(pos_shuffled, X, y_pos),
            "random_rep": evaluate_probe(pos_randrep, X, y_pos)
        },
        "DEP": {
            "linear": evaluate_probe(dep_model, X, y_dep),
            "nonlinear": evaluate_probe(dep_nonlinear, X, y_dep),
            "random": evaluate_probe(dep_random, X, y_dep),
            "shuffled": evaluate_probe(dep_shuffled, X, y_dep),
            "random_rep": evaluate_probe(dep_randrep, X, y_dep)
        },
        "POSITION": {
            "linear": evaluate_probe(position_model, X, y_position),
            "nonlinear": evaluate_probe(position_nonlinear, X, y_position),
            "random": evaluate_probe(position_random, X, y_position),
            "shuffled": evaluate_probe(position_shuffled, X, y_position),
            "random_rep": evaluate_probe(position_randrep, X, y_position)
        },
        "WORD": {
            "linear": evaluate_adaptive_probe(word_model, X, y_word),
            "nonlinear": evaluate_adaptive_probe(word_nonlinear, X, y_word),
            "random": 1.0 / len(le_word.classes_),  # Theoretical random performance
            "shuffled": evaluate_adaptive_probe(word_shuffled, X, y_word),
            "random_rep": evaluate_adaptive_probe(word_randrep, X, y_word)
        }
    }

    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print("\nResults:")
    for task, metrics in results.items():
        print(f"\n{task}:")
        for method, acc in metrics.items():
            print(f"  {method}: {acc:.4f}")

    return results


def load_model_from_info(model_info_path):
    """Load model and tokenizer from saved model info file."""
    try:
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        model_name = model_info['model_name']
        tokenizer = model_info['tokenizer']
        config = model_info['config']
        
        # Load model from the saved config and name
        model = AutoModel.from_pretrained(model_name, config=config)
        model.eval()
        
        return model, tokenizer, model_name
    except Exception as e:
        print(f"Error loading model info: {e}")
        return None, None, None


def load_data_from_csv(csv_path):
    """Load data from CSV file and associated embeddings and metadata."""
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Load embeddings
    embeddings_path = csv_path.replace('.csv', '_embeddings.pkl')
    with open(embeddings_path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    # Load metadata
    metadata_path = csv_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Add embeddings back to dataframe
    embeddings_list = [embeddings_dict[idx] for idx in df['embedding_idx']]
    df['embedding'] = embeddings_list
    
    return df, metadata


def main(args):
    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Output directory: {output_dir}")
    print(f"Loading embeddings from: {args.embeddings_path}")
    
    # Load dataset from CSV and associated files
    print(f"Loading data from: {args.embeddings_path}")
    df, metadata = load_data_from_csv(args.embeddings_path)
    
    # Try to load model info if requested
    if args.load_model:
        model_info_path = args.embeddings_path.replace('.csv', '_model_info.pkl')
        if os.path.exists(model_info_path):
            print(f"Loading model info from: {model_info_path}")
            model, tokenizer, model_name = load_model_from_info(model_info_path)
            if model is not None:
                print(f"Successfully loaded model: {model_name}")
                # Save model and tokenizer in output directory for reference
                with open(os.path.join(output_dir, "model_info.pkl"), 'wb') as f:
                    pickle.dump({'model_name': model_name, 'tokenizer': tokenizer, 'config': model.config}, f)
        else:
            print(f"Model info file not found: {model_info_path}")
    
    print(f"Dataset size: {len(df)} words")
    print(f"Model used for embeddings: {metadata.get('model_name', 'Unknown')}")
    print(f"Embedding dimension: {metadata.get('embedding_dim', df['embedding'].iloc[0].shape[0] if len(df) > 0 else None)}")
    
    # Prepare data for probing
    X = torch.stack(df['embedding'].tolist())
    
    le_pos = LabelEncoder().fit(df["pos"])
    le_dep = LabelEncoder().fit(df["dep"])
    le_word = LabelEncoder().fit(df["word"])
    
    y_pos = torch.tensor(le_pos.transform(df['pos'].values))
    y_dep = torch.tensor(le_dep.transform(df['dep'].values))
    y_word = torch.tensor(le_word.transform(df['word'].values))
    y_position = torch.tensor(df['position'].values)
    
    print(f"Number of unique POS tags: {len(le_pos.classes_)}")
    print(f"Number of unique dependencies: {len(le_dep.classes_)}")
    print(f"Vocabulary size: {len(le_word.classes_)}")
    print(f"Max position: {y_position.max().item()}")
    
    # Run probes
    if args.run_all_probes:
        run_all_probes_and_controls(X, y_pos, y_dep, y_position, y_word, le_pos, le_dep, le_word, output_dir, hidden_dim=args.hidden_dim)
    else:
        # Run only specific probes based on arguments
        if args.train_pos:
            print("\n=== Training POS Probe ===")
            if args.nonlinear:
                pos_model = train_probe(NonlinearProbe(X.shape[1], len(le_pos.classes_), hidden_dim=args.hidden_dim), 
                                        X, y_pos, "POS_Nonlinear", output_dir=output_dir)
                print(f"Nonlinear POS accuracy: {evaluate_probe(pos_model, X, y_pos):.4f}")
            else:
                pos_model = train_probe(LinearProbe(X.shape[1], len(le_pos.classes_)), 
                                        X, y_pos, "POS", output_dir=output_dir)
                print(f"POS accuracy: {evaluate_probe(pos_model, X, y_pos):.4f}")
        
        if args.train_dep:
            print("\n=== Training Dependency Probe ===")
            if args.nonlinear:
                dep_model = train_probe(NonlinearProbe(X.shape[1], len(le_dep.classes_), hidden_dim=args.hidden_dim), 
                                        X, y_dep, "DEP_Nonlinear", output_dir=output_dir)
                print(f"Nonlinear DEP accuracy: {evaluate_probe(dep_model, X, y_dep):.4f}")
            else:
                dep_model = train_probe(LinearProbe(X.shape[1], len(le_dep.classes_)), 
                                        X, y_dep, "DEP", output_dir=output_dir)
                print(f"DEP accuracy: {evaluate_probe(dep_model, X, y_dep):.4f}")
        
        if args.train_position:
            print("\n=== Training Position Probe ===")
            if args.nonlinear:
                position_model = train_probe(NonlinearProbe(X.shape[1], y_position.max().item() + 1, hidden_dim=args.hidden_dim), 
                                             X, y_position, "POSITION_Nonlinear", output_dir=output_dir)
                print(f"Nonlinear Position accuracy: {evaluate_probe(position_model, X, y_position):.4f}")
            else:
                position_model = train_probe(LinearProbe(X.shape[1], y_position.max().item() + 1), 
                                             X, y_position, "POSITION", output_dir=output_dir)
                print(f"Position accuracy: {evaluate_probe(position_model, X, y_position):.4f}")
        
        if args.train_word:
            print("\n=== Training Word Probe ===")
            if args.nonlinear:
                word_model = train_adaptive_probe(X, y_word, len(le_word.classes_), 
                                                  nonlinear=True, hidden_dim=args.hidden_dim, 
                                                  task_name="WORD_Nonlinear", output_dir=output_dir)
                print(f"Nonlinear word accuracy: {evaluate_adaptive_probe(word_model, X, y_word):.4f}")
                save_word_representations(word_model, X, y_word, le_word, output_dir)
            else:
                word_model = train_adaptive_probe(X, y_word, len(le_word.classes_), 
                                                  task_name="WORD", output_dir=output_dir)
                print(f"Word accuracy: {evaluate_adaptive_probe(word_model, X, y_word):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train probes on word-level embeddings")
    
    # Input parameters
    parser.add_argument("--embeddings_path", type=str, required=True,
                        help="Path to the embeddings file (.csv) generated by generate_word_embeddings.py")
    parser.add_argument("--load_model", action="store_true",
                        help="Load the model from the saved model info file")
    
    # Training parameters
    parser.add_argument("--run_all_probes", action="store_true",
                        help="Run all probes and control experiments")
    parser.add_argument("--train_pos", action="store_true",
                        help="Train POS probe")
    parser.add_argument("--train_dep", action="store_true",
                        help="Train dependency probe")
    parser.add_argument("--train_position", action="store_true",
                        help="Train position probe")
    parser.add_argument("--train_word", action="store_true",
                        help="Train word probe")
    parser.add_argument("--nonlinear", action="store_true",
                        help="Use nonlinear probes")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension for nonlinear probes")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./trained_models",
                        help="Directory to save trained models and results")
    
    args = parser.parse_args()
    
    # If no specific probes are selected, run all
    if not any([args.run_all_probes, args.train_pos, args.train_dep, args.train_position, args.train_word]):
        args.run_all_probes = True
    
    main(args)
