# Representation of Spatial Geometry

## Authors
- Vikram Natarajan
- Matthieu Tehenan
- Johnathan Michala
- Milton Lin

## Description

This repository contains the code and data for the paper "Representation of Spatial Geometry". The project focuses on generating and embedding sentences that describe spatial relationships between objects. The repository is organized into several key components:

1. **Data Generation**: The `generate.py` script generates sentences that describe spatial relationships between objects. It uses predefined objects and spatial relations to create a comprehensive set of sentences, which are then saved to a CSV file.

2. **Sentence Embeddings**: The `embed.py` script loads the generated sentences and computes their embeddings using various pre-trained models from the Sentence Transformers library. The embeddings are saved for further analysis and use.

3. **Data Storage**: The generated sentences and their embeddings are stored in the `data` directory. The `.gitignore` file ensures that large or unnecessary files are not tracked by Git.

## Repository Structure
