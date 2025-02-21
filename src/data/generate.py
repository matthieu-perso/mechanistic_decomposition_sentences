import itertools
import json
import csv
import os

def load_json_data(file_path, key):
    with open(file_path, 'r') as f:
        return json.load(f)[key]

objects = load_json_data('src/data/objects.json', "objects")
spatial_relations = load_json_data('src/data/relations.json', "spatial_relations")

relations = [relation for category in spatial_relations.values() for relation in category]
object_pairs = list(itertools.permutations(objects, 2))

# Creating sentences along with the relation and subject-object
sentences = [
    {
        'sentence': f"The {obj1} is {relation[0]} the {obj2}.",
        'relation': relation[0],
        'subject': obj1,
        'object': obj2
    }
    for obj1, obj2 in object_pairs
    for relation in relations
] + [
    {
        'sentence': f"The {obj2} is {relation[1]} the {obj1}.",
        'relation': relation[1],
        'subject': obj2,
        'object': obj1
    }
    for obj1, obj2 in object_pairs
    for relation in relations
]

# Save sentences to a CSV file
csv_file_path = 'generated_sentences.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['sentence', 'relation', 'subject', 'object'])
    writer.writeheader()
    for sentence_data in sentences:
        writer.writerow(sentence_data)

print(f"Total unique sentences generated: {len(sentences)}")
print(f"Sentences saved to {csv_file_path}")
