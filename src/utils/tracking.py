import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Try to import optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from huggingface_hub import HfApi, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

def init_wandb(
    project_name: str = "mechanistic-decomposition-sentence-embeddings",
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None
) -> None:
    """Initialize Weights & Biases run if API key is set."""
    if not WANDB_AVAILABLE or not os.getenv("WANDB_API_KEY"):
        return
        
    wandb.init(
        project=project_name,
        config=config,
        tags=tags,
        notes=notes
    )

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to W&B if configured."""
    if not WANDB_AVAILABLE or not wandb.run:
        return
        
    wandb.log(metrics, step=step)

def log_artifact(
    name: str,
    type: str,
    description: str,
    metadata: Optional[Dict[str, Any]] = None,
    path: Optional[Union[str, Path]] = None
) -> None:
    """Log an artifact to W&B if configured."""
    if not WANDB_AVAILABLE or not wandb.run:
        return
        
    artifact = wandb.Artifact(
        name=name,
        type=type,
        description=description,
        metadata=metadata
    )
    
    if path:
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
    
    wandb.log_artifact(artifact)

def upload_to_hub(
    model_path: Union[str, Path],
    dataset_path: Union[str, Path],
    metadata: Dict[str, Any],
    repo_id: str,
    private: bool = False
) -> None:
    """Upload model and dataset to Hugging Face Hub if token is set."""
    if not HF_AVAILABLE or not os.getenv("HF_TOKEN"):
        return
        
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, private=private, repo_type="model")
    except Exception:
        pass  # Repository might already exist
    
    # Upload model
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="model.pth",
        repo_id=repo_id
    )
    
    # Upload dataset
    api.upload_file(
        path_or_fileobj=str(dataset_path),
        path_in_repo="dataset.csv",
        repo_id=f"{repo_id}-dataset"
    )
    
    # Upload metadata
    api.upload_file(
        path_or_fileobj=json.dumps(metadata, indent=2),
        path_in_repo="metadata.json",
        repo_id=repo_id
    )

def finish_run() -> None:
    """Finish the current W&B run if configured."""
    if not WANDB_AVAILABLE or not wandb.run:
        return
        
    wandb.finish() 