#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Word Embedding Pipeline Orchestration Script
--------------------------------------------
This script orchestrates the full pipeline:
1. Generating word embeddings for multiple models
2. Running linguistic probes on the embeddings
3. Performing dictionary learning
"""

import os
import argparse
import subprocess
import yaml
import json
import gc
import psutil
import torch
from datetime import datetime
from pathlib import Path

from src.utils.tracking import init_wandb, log_metrics, log_artifact, finish_run, upload_to_hub

with open("config.yaml", "r") as file:
    MODELS = yaml.safe_load(file)["models"]


def run_command(command, verbose=True):
    """Run a shell command and optionally print output."""
    if verbose:
        print(f"Running: {' '.join(command)}")
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {' '.join(command)}")
        print(f"Error: {result.stderr}")
        return False
    
    if verbose:
        print(f"Command completed successfully")
        if result.stdout:
            print(result.stdout)
    
    return True


def generate_embeddings(model_name, args):
    """Generate word embeddings for a specific model."""
    # Create model-specific output path
    model_short_name = model_name.split('/')[-1]
    output_path = os.path.join(args.output_dir, f"{model_short_name}_embeddings.csv")
    
    # Check if embeddings already exist and skip_existing_embeddings is set
    if args.skip_existing_embeddings and os.path.exists(output_path):
        print(f"\n{'='*80}\nSkipping embedding generation for model: {model_name} (file already exists)\n{'='*80}")
        return output_path
    
    print(f"\n{'='*80}\nGenerating embeddings for model: {model_name}\n{'='*80}")
    
    # Build command
    command = [
        "python", os.path.join(args.src_dir, "generate_word_embeddings.py"),
        "--model_name", model_name,
        "--source", args.source,
        "--num_sentences", str(args.num_sentences),
        "--output_path", output_path,
        "--batch_size", str(args.embedding_batch_size)
    ]
    
    if args.download_resources:
        command.append("--download_resources")
    
    if args.verbose:
        command.append("--verbose")
    
    # Run command
    success = run_command(command, verbose=args.verbose)
    
    # Force garbage collection to free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if success:
        print(f"Embeddings generated successfully: {output_path}")
        return output_path
    else:
        print(f"Failed to generate embeddings for {model_name}")
        return None


def train_probes(embeddings_path, args):
    """Train linguistic probes on the generated embeddings."""
    print(f"\n{'='*80}\nTraining probes for: {embeddings_path}\n{'='*80}")
    
    # Create model-specific output directory
    model_short_name = os.path.basename(embeddings_path).replace("_embeddings.csv", "")
    probe_output_dir = os.path.join(args.output_dir, f"{model_short_name}_probes")
    
    # Build command
    command = [
        "python", os.path.join(args.src_dir, "train_word_probes.py"),
        "--embeddings_path", embeddings_path,
        "--output_dir", probe_output_dir,
        "--run_all_probes"
    ]
    
    if args.skip_word_probes:
        command.append("--skip_word_probes")
    
    if args.nonlinear_probes:
        command.append("--nonlinear")
        command.extend(["--hidden_dim", str(args.hidden_dim)])
    
    # Add Hugging Face Hub arguments if provided
    if args.hub_repo_id:
        command.extend(["--hub_repo_id", args.hub_repo_id])
        if args.hub_private:
            command.append("--hub_private")
    
    # Run command
    success = run_command(command, verbose=args.verbose)
    
    if success:
        print(f"Probes trained successfully: {probe_output_dir}")
        return probe_output_dir
    else:
        print(f"Failed to train probes for {embeddings_path}")
        return None


def run_dictionary_learning(embeddings_path, args):
    """Run dictionary learning on the generated embeddings."""
    print(f"\n{'='*80}\nRunning dictionary learning for: {embeddings_path}\n{'='*80}")
    
    # Create model-specific output directory
    model_short_name = os.path.basename(embeddings_path).replace("_embeddings.csv", "")
    dict_output_dir = os.path.join(args.output_dir, f"{model_short_name}_dict_learning")
    
    # Build command
    command = [
        "python", os.path.join(args.src_dir, "dictionary_learning.py"),
        "--embeddings_path", embeddings_path,
        "--output_dir", dict_output_dir,
        "--model_name", args.static_model,
        "--batch_size", str(args.dict_batch_size)
    ]
    
    if args.run_optuna:
        command.append("--run_optuna")
        command.extend(["--n_trials", str(args.n_trials)])
    
    if args.params_file:
        command.extend(["--params_file", args.params_file])
    
    if args.max_samples > 0:
        command.extend(["--max_samples", str(args.max_samples)])
    
    if args.no_cuda:
        command.append("--no_cuda")
    
    # Add Hugging Face Hub arguments if provided
    if args.hub_repo_id:
        command.extend(["--hub_repo_id", args.hub_repo_id])
        if args.hub_private:
            command.append("--hub_private")
    
    # Run command
    success = run_command(command, verbose=args.verbose)
    
    # Force garbage collection to free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if success:
        print(f"Dictionary learning completed successfully: {dict_output_dir}")
        return dict_output_dir
    else:
        print(f"Failed to run dictionary learning for {embeddings_path}")
        return None


def run_pipeline_for_model(model_name, args):
    """Run the complete pipeline for a single model."""
    print(f"\n{'#'*100}\nProcessing model: {model_name}\n{'#'*100}")
    
    # Print memory usage before starting
    print_memory_usage(f"Before processing model {model_name}")
    
    # Initialize W&B run for this model
    init_wandb(
        project_name=args.wandb_project,
        config={
            "model_name": model_name,
            **vars(args)
        },
        tags=args.wandb_tags + [model_name.split('/')[-1]],
        notes=args.wandb_notes or f"Running pipeline for model {model_name}"
    )
    
    # Step 1: Generate embeddings
    embeddings_path = generate_embeddings(model_name, args)
    if not embeddings_path:
        return None
    
    # Log embeddings generation
    log_metrics({
        "pipeline/embeddings_generated": 1,
        "pipeline/model": model_name
    })
    
    results = {
        "model_name": model_name,
        "embeddings_path": embeddings_path
    }
    
    # Print memory usage after embeddings generation
    print_memory_usage(f"After generating embeddings for {model_name}")
    
    # Step 2: Train probes
    if args.run_probes:
        probe_output_dir = train_probes(embeddings_path, args)
        results["probe_output_dir"] = probe_output_dir
        
        # Log probe results
        if probe_output_dir:
            log_metrics({
                "pipeline/probes_trained": 1
            })
            # Upload probe results to W&B with versioning
            model_short_name = model_name.split('/')[-1]
            log_artifact(
                name=f"probe-results-{model_short_name}",
                type="probe-results",
                description=f"Probe results for {model_name}",
                metadata=results,
                path=probe_output_dir,
                aliases=["latest", model_short_name, f"{model_short_name}-probes"]
            )
    
    # Step 3: Run dictionary learning
    if args.run_dict_learning:
        dict_output_dir = run_dictionary_learning(embeddings_path, args)
        results["dict_output_dir"] = dict_output_dir
        
        # Log dictionary learning results
        if dict_output_dir:
            log_metrics({
                "pipeline/dictionary_learning_completed": 1
            })
            # Upload dictionary learning results to W&B with versioning
            model_short_name = model_name.split('/')[-1]
            log_artifact(
                name=f"dict-learning-results-{model_short_name}",
                type="dict-learning-results",
                description=f"Dictionary learning results for {model_name}",
                metadata=results,
                path=dict_output_dir,
                aliases=["latest", model_short_name, f"{model_short_name}-dict-learning"]
            )
    
    # Upload to Hugging Face Hub if specified
    if args.hub_repo_id:
        model_short_name = model_name.split('/')[-1]
        repo_id = f"{args.hub_repo_id}-{model_short_name}"
        
        # Prepare metadata
        metadata = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": "1.0",
            "results": results
        }
        
        # Upload to HF Hub
        upload_to_hub(
            model_path=embeddings_path,
            dataset_path=os.path.dirname(embeddings_path),
            metadata=metadata,
            repo_id=repo_id,
            private=args.hub_private
        )
    
    # Log final results with versioning
    model_short_name = model_name.split('/')[-1]
    log_artifact(
        name=f"pipeline-results-{model_short_name}",
        type="results",
        description=f"Pipeline results for {model_name}",
        metadata=results,
        path=embeddings_path,
        aliases=["latest", model_short_name, f"{model_short_name}-complete"]
    )
    
    # Finish W&B run for this model
    finish_run()
    
    return results


def print_memory_usage(label="Current"):
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"\n{label} Memory Usage:")
    print(f"  RSS: {mem_info.rss / (1024 ** 2):.2f} MB")
    print(f"  VMS: {mem_info.vms / (1024 ** 2):.2f} MB")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"  CUDA Allocated: {allocated:.2f} MB")
        print(f"  CUDA Reserved: {reserved:.2f} MB")


def main(args):
    """Main function to run the pipeline for all models."""
    # Initialize W&B for the full experiment
    init_wandb(
        project_name=args.wandb_project,
        config=vars(args),
        tags=args.wandb_tags + ["experiment"],
        notes=args.wandb_notes or "Full pipeline experiment"
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    args_path = os.path.join(args.output_dir, "pipeline_args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print_memory_usage("Initial")
    
    # Process each model
    results = []
    for model_name in args.models:
        model_results = run_pipeline_for_model(model_name, args)
        if model_results:
            results.append(model_results)
        
        # Force garbage collection between models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_memory_usage(f"After processing {model_name}")
    
    # Save results
    results_path = os.path.join(args.output_dir, "pipeline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Log final results with versioning
    timestamp = datetime.now().strftime("%Y%m%d")
    log_artifact(
        name="pipeline-final-results",
        type="results",
        description="Final pipeline results",
        metadata=results,
        path=results_path,
        aliases=["latest", f"run-{timestamp}"]
    )
    
    print(f"\nPipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\nSummary:")
    for result in results:
        print(f"Model: {result['model_name']}")
        print(f"  Embeddings: {result.get('embeddings_path', 'Failed')}")
        print(f"  Probes: {result.get('probe_output_dir', 'Not run')}")
        print(f"  Dictionary Learning: {result.get('dict_output_dir', 'Not run')}")
        print()
    
    # Finish W&B run
    finish_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete word embedding pipeline")
    
    # Pipeline configuration
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="List of transformer models to process")
    parser.add_argument("--run_probes", action="store_true", default=True,
                        help="Run linguistic probes")
    parser.add_argument("--run_dict_learning", action="store_true", default=True,
                        help="Run dictionary learning")
    
    # Directory configuration
    parser.add_argument("--src_dir", type=str, default="./src",
                        help="Directory containing the source scripts")
    parser.add_argument("--output_dir", type=str, default="./pipeline_results",
                        help="Directory to save all results")
    
    # Embedding generation parameters
    parser.add_argument("--source", type=str, default="brown",
                        help="Source of sentences: 'brown' for Brown corpus or path to a text file")
    parser.add_argument("--num_sentences", type=int, default=20000,
                        help="Maximum number of sentences to process")
    parser.add_argument("--download_resources", action="store_true",
                        help="Download required NLTK and Stanza resources")
    parser.add_argument("--embedding_batch_size", type=int, default=32,
                        help="Batch size for processing sentences during embedding generation")
    
    # Probe parameters
    parser.add_argument("--nonlinear_probes", action="store_true",
                        help="Use nonlinear probes")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension for nonlinear probes")
    
    # Dictionary learning parameters
    parser.add_argument("--static_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Model to use for static embeddings in dictionary learning")
    parser.add_argument("--run_optuna", action="store_true",
                        help="Run Optuna hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=40,
                        help="Number of Optuna trials")
    parser.add_argument("--params_file", type=str, default="",
                        help="Path to JSON file with model parameters (if not running Optuna)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Maximum number of samples to use for dictionary learning (0 for all)")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--dict_batch_size", type=int, default=64,
                        help="Batch size for dictionary learning")
    parser.add_argument("--skip_word_probes", action="store_true",
                        help="Skip word prediction probes for faster training")
    parser.add_argument("--skip_existing_embeddings", action="store_true",
                        help="Skip embedding generation if output file already exists")
    
    # Misc parameters
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    
    # Add W&B arguments
    parser.add_argument("--wandb_project", type=str, default="mechanistic-decomposition-sentence-embeddings",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_tags", nargs="+", default=[],
                        help="Tags for W&B runs")
    parser.add_argument("--wandb_notes", type=str, default="",
                        help="Notes for the W&B run")
    
    # Add Hugging Face Hub arguments
    parser.add_argument("--hub_repo_id", type=str, default="",
        help="Hugging Face Hub repository ID (format: username/repo-name)")
    parser.add_argument("--hub_private", action="store_true",
        help="Make the Hugging Face Hub repository private")
    
    args = parser.parse_args()
    main(args)