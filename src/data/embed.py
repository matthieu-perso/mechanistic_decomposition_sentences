import torch
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

HF_DATASET = "matthieunlp/spatial_geometry"
DATA_DIR = "src/data/sentence-embeddings"

MODELS = [
    "Alibaba-NLP/gte-large-en-v1.5",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2"
]

def get_embeddings_path(model_name):
    return f"{DATA_DIR}/{model_name.replace('/', '_')}.pt"

print('Loading dataset...')
sentences = []
dataset = load_dataset(HF_DATASET, split="train")
for entry in tqdm(dataset):
    sentences.extend(entry.values())
print(f"Loaded {len(sentences)} sentences from dataset")

for model_name in MODELS:
    print(f"Processing with model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    embeddings = model.encode(sentences)
    print(embeddings.shape)

    data = [{"sentence": sentence, "embedding": embedding} 
            for sentence, embedding in zip(sentences, embeddings)]

    torch.save(data, get_embeddings_path(model_name))

    print(f"Saved embeddings to {DATA_DIR}/{model_name.replace('/', '_')}.pt\n")