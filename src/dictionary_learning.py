#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dictionary Learning Script
--------------------------
This script performs supervised dictionary learning for sentence decomposition.
It decomposes word embeddings into interpretable atoms with classification tasks.
"""

import os
import argparse
import json
import pickle
import gc
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import optuna


# ------------------------------
# Model Definition
# ------------------------------
class SupervisedDictionaryLearning(nn.Module):
    def __init__(self, d, k, num_pos, num_dep, nonlinearity='identity'):
        """
        Initialize the Supervised Dictionary Learning model.
        
        Args:
            d: Input dimension (embedding size)
            k: Dictionary size (number of atoms)
            num_pos: Number of POS tags
            num_dep: Number of dependency relations
            nonlinearity: Activation function ('identity' or 'relu')
        """
        super().__init__()
        self.S = nn.Linear(d, k, bias=False)   # Encoder (contextual to sparse code)
        self.D = nn.Linear(k, d, bias=False)   # Decoder (sparse code to contextual)

        # Supervised features
        self.pos_classifier = nn.Linear(k, num_pos)
        self.dep_classifier = nn.Linear(k, num_dep)

        # Optional nonlinearity
        if nonlinearity == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

    def forward(self, X):
        """
        Forward pass through the model.
        
        Args:
            X: Input tensor of shape (batch_size, d)
            
        Returns:
            x_rec: Reconstructed input
            S: Sparse codes
            pos_logits: POS tag logits
            dep_logits: Dependency relation logits
        """
        S = self.activation(self.S(X))
        x_rec = self.D(S)
        pos_logits = self.pos_classifier(S)
        dep_logits = self.dep_classifier(S)
        return x_rec, S, pos_logits, dep_logits


# ------------------------------
# Static Embeddings Extraction
# ------------------------------
def get_static_word_embeddings(df, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None, batch_size=100):
    """
    Extract static word embeddings for each unique word in the dataframe.
    
    Args:
        df: DataFrame containing a 'word' column
        model_name: Name of the transformer model to use
        device: Device to run the model on
        batch_size: Number of unique words to process at once
        
    Returns:
        DataFrame with added 'static_embedding' column
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to device if specified
    if device is not None:
        model = model.to(device)

    # Turn off gradients since we don't need backprop
    model.eval()
    
    # Get unique words
    unique_words = df['word'].unique()
    word_to_embedding = {}
    
    # Process words in batches to save memory
    num_batches = (len(unique_words) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Extracting static embeddings"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(unique_words))
            batch_words = unique_words[start_idx:end_idx]
            
            for word in batch_words:
                # Use encode method directly instead of tokenize + convert_tokens_to_ids
                # This avoids the TextEncodeInput type error
                try:
                    # Handle potential type errors with robust error handling
                    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False)
                    token_ids = inputs["input_ids"][0].tolist()
                except Exception as e:
                    print(f"Error tokenizing word '{word}': {e}")
                    token_ids = []
                
                token_embeddings = []

                for token_id in token_ids:
                    if token_id is not None and token_id < model.embeddings.word_embeddings.num_embeddings:
                        embedding = model.embeddings.word_embeddings.weight[token_id]
                        if device is not None:
                            embedding = embedding.to(device)
                        token_embeddings.append(embedding)

                # Pool all token embeddings for the word
                if token_embeddings:
                    pooled = torch.stack(token_embeddings, dim=0).mean(dim=0)  # mean pooling
                    word_to_embedding[word] = pooled.cpu()  # Move to CPU to save GPU memory
                else:
                    # Fallback if tokenization fails
                    word_to_embedding[word] = torch.zeros(model.config.hidden_size)
            
            # Force garbage collection after each batch
            if device is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Map pooled embeddings back to dataframe
    print("Mapping embeddings to dataframe...")
    df['static_embedding'] = df['word'].map(word_to_embedding)
    
    # Clean up to free memory
    del model, tokenizer, word_to_embedding
    if device is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return df


# ------------------------------
# Optuna Objective Function
# ------------------------------
def create_objective(X, y_pos, y_dep, word_static_tensor, num_pos, num_dep, device, batch_size=64):
    """
    Create an objective function for Optuna hyperparameter optimization.
    
    Args:
        X: Tensor of contextual embeddings
        y_pos: Tensor of POS tag labels
        y_dep: Tensor of dependency relation labels
        word_static_tensor: Tensor of static word embeddings
        num_pos: Number of POS tags
        num_dep: Number of dependency relations
        device: Device to run the model on
        batch_size: Batch size for training and validation (default: 64)
        
    Returns:
        objective: Objective function for Optuna
    """
    def objective(trial):
        # Make batch_size available in the inner function
        nonlocal batch_size
        # Hyperparams from trial
        d = X.shape[1]  # embedding dim
        k = trial.suggest_categorical("k", [64, 128])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        nonlin = trial.suggest_categorical("nonlinearity", ["identity", "relu"])

        # Weights for multi-objective
        alpha_pos = trial.suggest_float("alpha_pos", 0.0, 1.0)
        alpha_dep = trial.suggest_float("alpha_dep", 0.0, 1.0)
        alpha_static = trial.suggest_float("alpha_static", 0.0, 1.0)
        alpha_sparse = trial.suggest_float("alpha_sparse", 0.8, 1.0)

        # L1 lambdas
        l1_s_contextual = trial.suggest_float("l1_s_contextual", 1e-6, 1.0, log=True)
        l1_s_static = trial.suggest_float("l1_s_static", 1e-6, 1.0, log=True)

        model = SupervisedDictionaryLearning(d, k, num_pos, num_dep, nonlinearity=nonlin).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Base losses
        criterion_recon = nn.MSELoss()
        criterion_class = nn.CrossEntropyLoss()

        # Setup data with smaller batch sizes to avoid OOM
        dataset = TensorDataset(X, y_pos, y_dep, word_static_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        # Use the fixed batch size provided to the create_objective function
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Training loop
        model.train()
        for epoch in tqdm(range(40), desc=f"Trial {trial.number}"):
            for batch in train_loader:
                x_batch, y_p, y_d, w_static = [x.to(device) for x in batch]
                optimizer.zero_grad()

                # Forward pass for contextual
                x_rec, S_contextual, p_pred, d_pred = model(x_batch)

                # Basic losses
                loss_recon = criterion_recon(x_rec, x_batch)
                loss_pos = criterion_class(p_pred, y_p)
                loss_dep = criterion_class(d_pred, y_d)
                loss_sparse_contextual = l1_s_contextual * torch.norm(S_contextual, p=1)

                # Now handle static embedding
                x_rec_static, S_static, _, _ = model(w_static)
                loss_static_recon = criterion_recon(x_rec_static, w_static)
                loss_sparse_static = l1_s_static * torch.norm(S_static, p=1)

                # Weighted sum
                loss = (
                    loss_recon
                    + alpha_pos * loss_pos
                    + alpha_dep * loss_dep
                    + alpha_sparse * (loss_sparse_contextual + loss_sparse_static)
                    + alpha_static * loss_static_recon
                )

                loss.backward()
                optimizer.step()

        # ------------------------------
        # Validation pass for metrics
        # ------------------------------
        model.eval()

        val_recon_total = 0.0
        val_static_recon_total = 0.0
        val_pos_loss_total = 0.0
        val_dep_loss_total = 0.0
        val_s_contextual_total = 0.0
        val_s_static_total = 0.0
        val_samples = 0

        # For classification
        y_true_pos, y_pred_pos = [], []
        y_true_dep, y_pred_dep = [], []

        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_p, y_d, w_static = [x.to(device) for x in batch]
                batch_size = x_batch.size(0)
                val_samples += batch_size

                # 1) Contextual forward
                x_rec, S_contextual, p_pred, d_pred = model(x_batch)
                val_recon_total += criterion_recon(x_rec, x_batch).item() * batch_size
                val_pos_loss_total += criterion_class(p_pred, y_p).item() * batch_size
                val_dep_loss_total += criterion_class(d_pred, y_d).item() * batch_size
                val_s_contextual_total += torch.norm(S_contextual, p=1).item()

                # 2) Static forward
                x_rec_static, S_static, _, _ = model(w_static)
                val_static_recon_total += criterion_recon(x_rec_static, w_static).item() * batch_size
                val_s_static_total += torch.norm(S_static, p=1).item()

                # Save classification for F1
                y_true_pos.append(y_p.cpu())
                y_pred_pos.append(p_pred.argmax(dim=1).cpu())

                y_true_dep.append(y_d.cpu())
                y_pred_dep.append(d_pred.argmax(dim=1).cpu())

        # Averages
        mean_val_recon = val_recon_total / val_samples
        mean_val_static_recon = val_static_recon_total / val_samples
        mean_val_pos_loss = val_pos_loss_total / val_samples
        mean_val_dep_loss = val_dep_loss_total / val_samples
        mean_val_s_contextual = val_s_contextual_total / val_samples
        mean_val_s_static = val_s_static_total / val_samples

        # F1
        y_true_pos = torch.cat(y_true_pos).numpy()
        y_pred_pos = torch.cat(y_pred_pos).numpy()
        f1_pos = f1_score(y_true_pos, y_pred_pos, average='macro')

        y_true_dep = torch.cat(y_true_dep).numpy()
        y_pred_dep = torch.cat(y_pred_dep).numpy()
        f1_dep = f1_score(y_true_dep, y_pred_dep, average='macro')

        # Log them as user attrs in Optuna
        trial.set_user_attr('val_recon', mean_val_recon)
        trial.set_user_attr('val_static_recon', mean_val_static_recon)
        trial.set_user_attr('val_pos_loss', mean_val_pos_loss)
        trial.set_user_attr('val_dep_loss', mean_val_dep_loss)
        trial.set_user_attr('val_s_contextual', mean_val_s_contextual)
        trial.set_user_attr('val_s_static', mean_val_s_static)
        trial.set_user_attr('f1_pos', f1_pos)
        trial.set_user_attr('f1_dep', f1_dep)

        # Score to maximize
        score = f1_pos + f1_dep

        print(f"F1 POS: {f1_pos:.4f}, F1 DEP: {f1_dep:.4f}, Recon: {mean_val_recon:.4f}, "
              f"Static Recon: {mean_val_static_recon:.4f}, POS Loss: {mean_val_pos_loss:.4f}, "
              f"DEP Loss: {mean_val_dep_loss:.4f}")

        return score
    
    return objective


# ------------------------------
# Train Final Model
# ------------------------------
def train_final_model(X, y_pos, y_dep, word_static_tensor, num_pos, num_dep, best_params, device, output_dir):
    """
    Train the final model with the best hyperparameters.
    
    Args:
        X: Tensor of contextual embeddings
        y_pos: Tensor of POS tag labels
        y_dep: Tensor of dependency relation labels
        word_static_tensor: Tensor of static word embeddings
        num_pos: Number of POS tags
        num_dep: Number of dependency relations
        best_params: Best hyperparameters from Optuna study
        device: Device to run the model on
        output_dir: Directory to save the model and results
        
    Returns:
        model: Trained model
        results: Dictionary of evaluation results
    """
    # Extract hyperparameters
    k = best_params['k']
    lr = best_params['lr']
    nonlinearity = best_params['nonlinearity']
    alpha_pos = best_params['alpha_pos']
    alpha_dep = best_params['alpha_dep']
    alpha_static = best_params['alpha_static']
    alpha_sparse = best_params['alpha_sparse']
    l1_s_contextual = best_params['l1_s_contextual']
    l1_s_static = best_params['l1_s_static']

    # Model setup
    d = X.shape[1]  # Feature dimension (input)
    model = SupervisedDictionaryLearning(d, k, num_pos, num_dep, nonlinearity=nonlinearity).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    # Split data with smaller batch sizes to avoid OOM
    dataset = TensorDataset(X, y_pos, y_dep, word_static_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Use the provided batch size instead of hardcoded values
    batch_size = best_params.get('batch_size', 64)  # Use from best_params if available, otherwise default to 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    print(f"Using batch size: {batch_size}")

    # Training loop
    model.train()
    for epoch in tqdm(range(40), desc=f'Training Final Model'):
        for batch in train_loader:
            x_batch, y_p, y_d, w_static = batch
            x_batch, y_p, y_d, w_static = x_batch.to(device), y_p.to(device), y_d.to(device), w_static.to(device)
            optimizer.zero_grad()

            # Forward pass for contextual
            x_rec, S_contextual, p_pred, d_pred = model(x_batch)

            # Basic losses
            loss_recon = criterion_recon(x_rec, x_batch)
            loss_pos = criterion_class(p_pred, y_p)
            loss_dep = criterion_class(d_pred, y_d)
            loss_sparse_contextual = l1_s_contextual * torch.norm(S_contextual, p=1)

            # Now handle static embedding
            x_rec_static, S_static, _, _ = model(w_static)
            loss_static_recon = criterion_recon(x_rec_static, w_static)
            loss_sparse_static = l1_s_static * torch.norm(S_static, p=1)

            # Weighted sum
            loss = (
                loss_recon
                + alpha_pos * loss_pos
                + alpha_dep * loss_dep
                + alpha_sparse * (loss_sparse_contextual + loss_sparse_static)
                + alpha_static * loss_static_recon
            )

            loss.backward()
            optimizer.step()

    # Validation Metrics
    model.eval()

    val_recon_total = 0.0
    val_static_recon_total = 0.0
    val_pos_loss_total = 0.0
    val_dep_loss_total = 0.0
    val_s_contextual_total = 0.0
    val_s_static_total = 0.0
    val_samples = 0

    # For classification
    y_true_pos, y_pred_pos = [], []
    y_true_dep, y_pred_dep = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            x_batch, y_p, y_d, w_static = [x.to(device) for x in batch]
            batch_size = x_batch.size(0)
            val_samples += batch_size

            # 1) Contextual forward
            x_rec, S_contextual, p_pred, d_pred = model(x_batch)
            val_recon_total += criterion_recon(x_rec, x_batch).item() * batch_size
            val_pos_loss_total += criterion_class(p_pred, y_p).item() * batch_size
            val_dep_loss_total += criterion_class(d_pred, y_d).item() * batch_size
            val_s_contextual_total += torch.norm(S_contextual, p=1).item()

            # 2) Static forward
            x_rec_static, S_static, _, _ = model(w_static)
            val_static_recon_total += criterion_recon(x_rec_static, w_static).item() * batch_size
            val_s_static_total += torch.norm(S_static, p=1).item()

            # Save classification for F1
            y_true_pos.extend(y_p.cpu().numpy())
            y_pred_pos.extend(p_pred.argmax(dim=1).cpu().numpy())

            y_true_dep.extend(y_d.cpu().numpy())
            y_pred_dep.extend(d_pred.argmax(dim=1).cpu().numpy())

    # Averages
    mean_val_recon = val_recon_total / val_samples
    mean_val_static_recon = val_static_recon_total / val_samples
    mean_val_pos_loss = val_pos_loss_total / val_samples
    mean_val_dep_loss = val_dep_loss_total / val_samples
    mean_val_s_contextual = val_s_contextual_total / val_samples
    mean_val_s_static = val_s_static_total / val_samples

    # F1
    f1_pos = f1_score(y_true_pos, y_pred_pos, average='macro')
    f1_dep = f1_score(y_true_dep, y_pred_dep, average='macro')

    print(f"Validation F1 Score (POS): {f1_pos:.4f}")
    print(f"Validation F1 Score (Dep): {f1_dep:.4f}")

    # Save results
    results = {
        'f1_pos': f1_pos,
        'f1_dep': f1_dep,
        'mean_val_recon': mean_val_recon,
        'mean_val_static_recon': mean_val_static_recon,
        'mean_val_pos_loss': mean_val_pos_loss,
        'mean_val_dep_loss': mean_val_dep_loss,
        'mean_val_s_contextual': mean_val_s_contextual,
        'mean_val_s_static': mean_val_s_static,
    }

    return model, results


def main(args):
    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"dict_learning_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}", flush=True)
    
    # Save arguments
    args_path = os.path.join(output_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved arguments to: {args_path}", flush=True)
    
    print(f"Output directory: {output_dir}", flush=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # Load data
    print(f"Loading data from: {args.embeddings_path}", flush=True)
    df = pd.read_csv(args.embeddings_path)
    print(f"Loaded CSV with {len(df)} rows", flush=True)
    
    # Limit dataset size if specified
    if args.max_samples > 0:
        print(f"Limiting dataset to {args.max_samples} samples", flush=True)
        df = df.sample(min(args.max_samples, len(df)), random_state=42) if len(df) > args.max_samples else df
        print(f"Dataset size after limiting: {len(df)} samples", flush=True)
    
    # Load embeddings in chunks to save memory
    print(f"Loading embeddings from: {args.embeddings_path.replace('.csv', '_embeddings.pkl')}", flush=True)
    embeddings_path = args.embeddings_path.replace('.csv', '_embeddings.pkl')
    with open(embeddings_path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    print(f"Loaded embeddings dictionary with {len(embeddings_dict)} entries", flush=True)
    
    # Load metadata
    print(f"Loading metadata from: {args.embeddings_path.replace('.csv', '_metadata.json')}", flush=True)
    metadata_path = args.embeddings_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded metadata: {metadata}", flush=True)
    
    # Add embeddings back to dataframe in batches
    print("Adding embeddings to dataframe...", flush=True)
    batch_size = 1000  # Process in smaller chunks to avoid OOM
    all_embeddings = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, len(df), batch_size), 1):
        batch_end = min(i+batch_size, len(df))
        print(f"Processing batch {batch_num}/{total_batches}: rows {i} to {batch_end-1}", flush=True)
        batch_df = df.iloc[i:batch_end]
        batch_embeddings = [embeddings_dict[idx] for idx in batch_df['embedding_idx']]
        all_embeddings.extend(batch_embeddings)
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Completed batch {batch_num}/{total_batches}", flush=True)
    
    df['embedding'] = all_embeddings
    
    # Extract static embeddings if not already present
    if 'static_embedding' not in df.columns:
        print("Extracting static embeddings...")
        df = get_static_word_embeddings(df, model_name=args.model_name, device=device, batch_size=args.batch_size)
    
    # Prepare data for dictionary learning
    print("Preparing data for dictionary learning...")
    
    # Encode labels first to save memory
    print("Encoding labels...")
    le_pos = LabelEncoder().fit(df["pos"])
    le_dep = LabelEncoder().fit(df["dep"])
    
    y_pos = torch.tensor(le_pos.transform(df['pos'].values))
    y_dep = torch.tensor(le_dep.transform(df['dep'].values))
    
    # Convert embeddings to tensors in batches
    print("Converting embeddings to tensors...")
    X_batches = []
    static_batches = []
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        X_batches.append(torch.stack(batch_df['embedding'].tolist()))
        static_batches.append(torch.stack(batch_df['static_embedding'].tolist()))
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    X = torch.cat(X_batches)
    word_static_tensor = torch.cat(static_batches)
    
    # Clean up to free memory
    del all_embeddings, X_batches, static_batches, embeddings_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    num_pos = df['pos'].nunique()
    num_dep = df['dep'].nunique()
    
    print(f"Number of samples: {len(df)}")
    print(f"Number of unique POS tags: {num_pos}")
    print(f"Number of unique dependencies: {num_dep}")
    print(f"Embedding dimension: {X.shape[1]}")
    
    # Save label encoders
    with open(os.path.join(output_dir, "le_pos.pkl"), 'wb') as f:
        pickle.dump(le_pos, f)
    with open(os.path.join(output_dir, "le_dep.pkl"), 'wb') as f:
        pickle.dump(le_dep, f)
    
    # Run Optuna study if requested
    if args.run_optuna:
        print("Running Optuna hyperparameter optimization...")
        
        # Use a smaller sample for Optuna to make hyperparameter optimization more efficient
        optuna_sample_size = min(10000, len(X))  # Use first 10000 samples for Optuna or less if dataset is smaller
        
        if len(X) > optuna_sample_size:
            print(f"Using first {optuna_sample_size} samples for Optuna hyperparameter optimization (out of {len(X)} total samples)")
            # Use the first N entries to keep contiguous sentences together
            X_optuna = X[:optuna_sample_size]
            y_pos_optuna = y_pos[:optuna_sample_size]
            y_dep_optuna = y_dep[:optuna_sample_size]
            word_static_optuna = word_static_tensor[:optuna_sample_size]
        else:
            # If dataset is already small, use all of it
            print(f"Using all {len(X)} samples for Optuna hyperparameter optimization")
            X_optuna = X
            y_pos_optuna = y_pos
            y_dep_optuna = y_dep
            word_static_optuna = word_static_tensor
        
        objective = create_objective(X_optuna, y_pos_optuna, y_dep_optuna, word_static_optuna, 
                                    num_pos, num_dep, device, batch_size=args.batch_size)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.n_trials)
        
        # Save study
        with open(os.path.join(output_dir, "optuna_study.pkl"), 'wb') as f:
            pickle.dump(study, f)
        
        # Save best parameters
        with open(os.path.join(output_dir, "best_params.json"), 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        best_params = study.best_params
    elif args.params_file:
        # Load parameters from file
        print(f"Loading parameters from: {args.params_file}")
        with open(args.params_file, 'r') as f:
            best_params = json.load(f)
    else:
        # Use default parameters
        print("Using default parameters")
        best_params = {
            'k': 128,
            'lr': 0.001,
            'nonlinearity': 'relu',
            'alpha_pos': 0.2,
            'alpha_dep': 0.1,
            'alpha_static': 0.5,
            'alpha_sparse': 0.9,
            'l1_s_contextual': 0.0001,
            'l1_s_static': 0.0001
        }
    
    # Train final model
    print("Training final model...")
    model, results = train_final_model(
        X, y_pos, y_dep, word_static_tensor, num_pos, num_dep, best_params, device, output_dir
    )
    
    # Save model and results
    model_path = os.path.join(output_dir, "model.pth")
    results_path = os.path.join(output_dir, "results.json")
    
    print(f"Saving model to: {model_path}", flush=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully", flush=True)
    
    print(f"Saving results to: {results_path}", flush=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved successfully", flush=True)
    
    print(f"Model and results saved to: {output_dir}", flush=True)
    print("Results:", flush=True)
    for key, value in results.items():
        print(f"  {key}: {value}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dictionary Learning Script")
    
    # Input parameters
    parser.add_argument("--embeddings_path", type=str, required=True,
                        help="Path to the embeddings file (.csv) generated by generate_word_embeddings.py")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Transformer model name to use for static embeddings")
    
    # Training parameters
    parser.add_argument("--run_optuna", action="store_true",
                        help="Run Optuna hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=40,
                        help="Number of Optuna trials")
    parser.add_argument("--params_file", type=str, default="",
                        help="Path to JSON file with model parameters (if not running Optuna)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Maximum number of samples to use (0 for all)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training and validation")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./dict_learning_models",
                        help="Directory to save trained models and results")
    
    # Misc parameters
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    main(args)