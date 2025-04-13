#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download Artifacts Script
------------------------
This script downloads artifacts from Weights & Biases.
It can be used to retrieve embeddings and other artifacts from previous runs.
"""

import os
import argparse
import wandb
from pathlib import Path


def download_artifact(artifact_name, artifact_type, version="latest", project=None, entity=None, output_dir="./downloads"):
    """
    Download an artifact from Weights & Biases.
    
    Args:
        artifact_name: Name of the artifact to download
        artifact_type: Type of the artifact (e.g., 'embeddings', 'model', etc.)
        version: Version of the artifact to download (default: "latest")
        project: W&B project name
        entity: W&B entity name
        output_dir: Directory to save the downloaded artifact
    
    Returns:
        Path to the downloaded artifact
    """
    print(f"Downloading artifact: {artifact_name} (version: {version})")
    
    # Initialize W&B API
    api = wandb.Api()
    
    # Construct the artifact path
    if entity and project:
        artifact_path = f"{entity}/{project}/{artifact_name}:{version}"
    elif project:
        artifact_path = f"{project}/{artifact_name}:{version}"
    else:
        artifact_path = f"{artifact_name}:{version}"
    
    try:
        # Get the artifact
        artifact = api.artifact(artifact_path, type=artifact_type)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download the artifact
        download_dir = artifact.download(root=output_dir)
        print(f"Artifact downloaded to: {download_dir}")
        return download_dir
    
    except Exception as e:
        print(f"Error downloading artifact: {e}")
        return None


def list_artifacts(project=None, entity=None, artifact_type=None):
    """
    List all artifacts in a W&B project.
    
    Args:
        project: W&B project name
        entity: W&B entity name
        artifact_type: Filter by artifact type
    """
    print(f"Listing artifacts for project: {project or 'all'}")
    
    # Initialize W&B API
    api = wandb.Api()
    
    # Construct the project path
    if entity and project:
        project_path = f"{entity}/{project}"
    elif project:
        project_path = project
    else:
        print("Please specify a project name")
        return
    
    try:
        # Get the project
        project = api.project(project_path)
        
        # Get artifacts
        artifacts = api.artifacts(project_path, type=artifact_type)
        
        print(f"\nFound {len(artifacts)} artifacts:")
        for artifact in artifacts:
            print(f"  - {artifact.name} (type: {artifact.type}, version: {artifact.version})")
            
            # Print aliases if available
            if artifact.aliases:
                print(f"    Aliases: {', '.join(artifact.aliases)}")
    
    except Exception as e:
        print(f"Error listing artifacts: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download artifacts from Weights & Biases")
    
    # Command options
    parser.add_argument("--list", action="store_true", help="List available artifacts")
    parser.add_argument("--download", action="store_true", help="Download an artifact")
    
    # W&B options
    parser.add_argument("--entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--project", type=str, default="mechanistic-decomposition-sentence-embeddings", 
                        help="W&B project name")
    
    # Artifact options
    parser.add_argument("--name", type=str, help="Artifact name to download")
    parser.add_argument("--type", type=str, default="embeddings", 
                        help="Artifact type (e.g., 'embeddings', 'model')")
    parser.add_argument("--version", type=str, default="latest", 
                        help="Artifact version to download (default: latest)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="./downloads", 
                        help="Directory to save downloaded artifacts")
    
    args = parser.parse_args()
    
    if args.list:
        list_artifacts(project=args.project, entity=args.entity, artifact_type=args.type)
    
    elif args.download:
        if not args.name:
            print("Please specify an artifact name to download using --name")
            return
        
        download_artifact(
            artifact_name=args.name,
            artifact_type=args.type,
            version=args.version,
            project=args.project,
            entity=args.entity,
            output_dir=args.output_dir
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
