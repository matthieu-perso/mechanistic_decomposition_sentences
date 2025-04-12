# Mechanistic Decomposition of Sentence Representations

In this paper, we provide a method to mechanistically decompose transformer-based sentence embedding models. The project includes linguistic probing, dictionary learning, and pooling analysis.

## Project Structure

- `src/`: Source code directory containing core functionality
- `pipeline_results/`: Directory for storing experiment results
- `run_pipeline.py`: Main script for running the full analysis pipeline
- `config.yaml`: Configuration file for model specifications
- Jupyter notebooks for visualization:
  - `1_probes.ipynb`: Linguistic probing experiments
  - `2_dictionary.ipynb`: Dictionary learning analysis
  - `3_pooling.ipynb`: Pooling mechanism analysis

## Setup

This project uses Poetry for dependency management. To set up the environment:

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone this repository and install dependencies:
   ```bash
   git clone [repository-url]
   cd spatial_geometry
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

4. (Optional) To use Weights & Biases or Hugging Face Hub, set your API tokens:
   ```bash
   export WANDB_API_KEY="your-wandb-key"
   export HF_TOKEN="your-hf-token"
   ```

## Usage

The main analysis pipeline can be run using the `run_pipeline.py` script:

```bash
python run_pipeline.py \
    --models "sentence-transformers/all-MiniLM-L6-v2" \
    --run_probes \
    --run_dict_learning
```

If you have set your API tokens, you can add these optional arguments:
```bash
python run_pipeline.py \
    --models "sentence-transformers/all-MiniLM-L6-v2" \
    --run_probes \
    --run_dict_learning \
    --hub_repo_id "matthieunlp/mechanistic-decomposition-sentence-embeddings" \
    --wandb_project "mechanistic-decomposition-sentence-embeddings" \
    --wandb_tags "experiment" "v1"
```

The pipeline performs the following steps:
1. Generates word embeddings for specified models
2. Runs linguistic probes on the embeddings
3. Performs dictionary learning analysis
4. Analyzes pooling mechanisms

### Configuration

Edit `config.yaml` to specify:
- Models to analyze
- Dataset configurations
- Experiment parameters

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

