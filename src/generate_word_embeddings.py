#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Word Embeddings Script
-------------------------------
This script processes sentences and generates word-level embeddings aligned with
linguistic properties like part of speech and dependencies.
"""

import os
import json
import pickle
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import nltk
import stanza
import pandas as pd
from nltk.corpus import brown
from transformers import AutoTokenizer, AutoModel


def reconstruct_sentence(tokens):
    """Reconstruct a sentence from tokens."""
    sentence = " ".join(tokens)
    sentence = sentence.replace('``', '').replace("''", "").replace(
        " ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    return sentence


def get_word_embeddings_aligned(sentence, tokenizer, model, nlp):
    """
    Given a sentence, aligns subword embeddings from transformer model to words using char offsets from Stanza.
    Returns a list of dicts with word, embedding, POS, dependency, and position.
    """
    doc = nlp(sentence)
    word_spans = [(word.text, word.start_char, word.end_char, word.upos, word.deprel) 
                  for sent in doc.sentences for word in sent.words]

    # Tokenize with offset mapping, no special tokens
    encoding = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=False
    )
    offsets = encoding["offset_mapping"][0].tolist()
    input_ids = encoding["input_ids"]

    # Get subword embeddings
    with torch.no_grad():
        output = model(**{k: v for k, v in encoding.items() if k != 'offset_mapping'})
        subword_embeddings = output.last_hidden_state.squeeze(0)  # [seq_len, dim]

    # Align subwords to words
    aligned_data = []
    for i, (word, w_start, w_end, upos, deprel) in enumerate(word_spans):
        matching_sub_idxs = [j for j, (s, e) in enumerate(offsets) if s < w_end and e > w_start and s != e]

        if matching_sub_idxs:
            embs = [subword_embeddings[j] for j in matching_sub_idxs]
            word_embedding = torch.stack(embs).mean(dim=0)
            aligned_data.append({
                "word": word,
                "embedding": word_embedding,
                "pos": upos,
                "dep": deprel,
                "position": i
            })

    return aligned_data


def main(args):
    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"Using model: {args.model_name}")
    
    # Download required resources
    if args.download_resources:
        print("Downloading required resources...")
        nltk.download('brown')
        stanza.download('en')
    
    # Load stanza pipeline
    print("Loading Stanza pipeline...")
    nlp = stanza.Pipeline('en', processors='tokenize,pos,depparse,lemma')
    
    # Load transformer model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.eval()
    
    # Get sentences from source
    if args.source == 'brown':
        print("Getting sentences from Brown corpus...")
        sentences = [reconstruct_sentence(tokens) for tokens in brown.sents()]
        sentences = sentences[:args.num_sentences]
    else:
        print(f"Reading sentences from file: {args.source}")
        with open(args.source, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        sentences = sentences[:args.num_sentences]
    
    print(f"Processing {len(sentences)} sentences...")
    
    # Process sentences to get aligned word embeddings
    all_rows = []
    for i, sent in tqdm(enumerate(sentences), total=len(sentences), desc="Processing sentences"):
        try:
            aligned = get_word_embeddings_aligned(sent, tokenizer, model, nlp)
            for row in aligned:
                row["sentence_id"] = i
                row["sentence"] = sent
                all_rows.append(row)
        except Exception as e:
            if args.verbose:
                print(f"Error processing sentence {i}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_rows)
    
    # Create metadata dictionary
    metadata = {
        'model_name': args.model_name,
        'embedding_dim': df['embedding'].iloc[0].shape[0] if len(df) > 0 else None,
        'generation_timestamp': timestamp
    }
    
    # Extract embeddings before saving to CSV
    embeddings = {}
    for idx, row in enumerate(df.itertuples()):
        embeddings[idx] = row.embedding
    
    # Save embeddings separately
    embeddings_path = args.output_path.replace('.csv', '_embeddings.pkl')
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Save metadata separately
    metadata_path = args.output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save model configuration and tokenizer for later use
    model_save_path = args.output_path.replace('.csv', '_model_info.pkl')
    with open(model_save_path, 'wb') as f:
        model_info = {
            'model_name': args.model_name,
            'tokenizer': tokenizer,
            'config': model.config
        }
        pickle.dump(model_info, f)
    
    # Remove embedding column before saving to CSV
    df_csv = df.copy()
    df_csv['embedding_idx'] = range(len(df_csv))  # Add index to link back to embeddings
    df_csv = df_csv.drop(columns=['embedding'])
    
    # Save dataframe to CSV
    df_csv.to_csv(args.output_path, index=False)
    
    print(f"Dataset saved to {args.output_path}")
    print(f"Embeddings saved to {embeddings_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Model info saved to {model_save_path}")
    print(f"Dataset size: {len(df)} words from {len(sentences)} sentences")
    print(f"Embedding dimension: {metadata['embedding_dim']}")
    
    # Print some statistics
    if len(df) > 0:
        print(f"Number of unique POS tags: {df['pos'].nunique()}")
        print(f"Number of unique dependencies: {df['dep'].nunique()}")
        print(f"Vocabulary size: {df['word'].nunique()}")
        print(f"Max position: {df['position'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate word-level embeddings from sentences")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Transformer model name to use")
    
    # Input parameters
    parser.add_argument("--source", type=str, default="brown",
                        help="Source of sentences: 'brown' for Brown corpus or path to a text file")
    parser.add_argument("--num_sentences", type=int, default=20000,
                        help="Maximum number of sentences to process")
    
    # Output parameters
    parser.add_argument("--output_path", type=str, default="./embeddings.csv",
                        help="Path to save the processed embeddings as CSV")
    
    # Misc parameters
    parser.add_argument("--download_resources", action="store_true",
                        help="Download required NLTK and Stanza resources")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    
    args = parser.parse_args()
    main(args)
