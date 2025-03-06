import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download

class EmbeddingProcessor:
    def __init__(self, repo_id, filename, dataset_name, split='train'):
        self.repo_id = repo_id
        self.filename = filename
        self.dataset_name = dataset_name
        self.split = split
        self.data_base = None
        self.df = None
        self.inverted_sentence_pairs = None

    def download_data(self):
        file_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            repo_type="dataset"
        )
        self.data_base = torch.load(file_path)
        print(f"Length of the object: {len(self.data_base)}")

    def load_and_merge_dataset(self):
        dataset = load_dataset(self.dataset_name, split=self.split)
        self.df = pd.DataFrame(dataset)
        self.df = self.df.merge(pd.DataFrame(self.data_base), on=['sentence', 'relation', 'subject', 'object'])
        print(self.df.columns)
        print(self.df['subject'])

    def create_canonical_key(self):
        self.df["canonical_key"] = self.df.apply(
            lambda row: tuple(sorted([row["subject"], row["object"]])),
            axis=1
        )

    def filter_inverted_pairs(self):
        inverted_pairs = self.df.groupby(["relation", "canonical_key"]).filter(lambda group: len(group) > 1)
        self.inverted_sentence_pairs = (
            inverted_pairs.groupby(["relation", "canonical_key"]).apply(lambda group: pd.DataFrame({
                "sentences": [tuple(group["sentence"])],
                "embeddings": [tuple(group["embedding"])]
            })).reset_index()
        )
        print("BASE", self.inverted_sentence_pairs.columns)

    def calculate_embedding_differences(self):
        self.inverted_sentence_pairs["embedding_difference_1"] = self.inverted_sentence_pairs.apply(
            lambda row: torch.tensor(row["embeddings"][0]) - torch.tensor(row["embeddings"][1]), axis=1
        )
        self.inverted_sentence_pairs["embedding_difference_2"] = self.inverted_sentence_pairs.apply(
            lambda row: torch.tensor(row["embeddings"][1]) - torch.tensor(row["embeddings"][0]), axis=1
        )