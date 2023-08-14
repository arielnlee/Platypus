from datasets import load_dataset
import re
import json
import requests

# Trying to load the data directly from huggingface was giving errors, so I manually downloaded it from here:
#  https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.4.1/tree/main; 

dataset = load_dataset("jondurbin/airoboros-gpt4-1.4.1", data_files="instructions.jsonl")

data = dataset['train']

# Filter based on words in instruction

# Define a list of math and STEM-related keywords
keywords_expanded = [
    # Mathematics
    "algebra", "geometry", "calculus", "statistics", "probability", "theorem", "proof", "equation", 
    "integral", "derivative", "matrix", "vector", "graph", "function", "complex number", "real number", 
    "imaginary number", "differential", "fraction", "decimal", "percent", "logarithm", 
    "sequence", "sum", "product", "difference", "quotient", "prime", "composite", "factorial", "binomial", 
    "polynomial", "exponential", "pi", "euler", "infinity", "limit", "derivative", "integral", 
    "differential equation", "linear algebra", "set theory", "group theory", "ring theory", "field theory", 
    "number theory", "combinatorics", "topology", "measure theory", "game theory", "cryptology", 
    "algorithm", "computation","percentage","calculation"
    
    # General STEM
    "science", "technology", "engineering", "physics", "chemistry", "biology", "computer science", 
    "information technology", "environmental", "aerospace", 
    "biomedical", "chemical", "robotics", "AI", "artificial intelligence", 
    "machine learning", "deep learning", "neural network", "algorithm", "programming", "coding", 
    "software", "hardware", "network", "database", "security", "cybersecurity", "blockchain", 
    "virtual reality", "augmented reality", "quantum", "nanotechnology", "biotechnology", "genetics", 
    "genomics", "solar", "wind", "hydro", 
    "nuclear", "fossil fuel", "carbon", "greenhouse gas", "pollution", "conservation", "biodiversity", 
    "ecosystem", "species", "evolution", "cell", "molecule", "atom", "particle", "quantum", "gravity", 
    "relativity", "momentum", "velocity", "acceleration", "mass", "heat", "light", "sound", "electricity", "magnetism",
]

# Filter the data based on the presence of any of these keywords as whole words in the instruction field
filtered_data_keywords = [item for item in data if any(re.search(r'\b' + keyword + r'\b', item['instruction'].lower()) for keyword in keywords_expanded)]

# remove the 'category' and 'question_id' fields
for item in filtered_data_keywords:
    del item['category']
    del item['question_id']

# rename 'response' to output, and and an 'input' field which is empty after instrcution and before output

for item in filtered_data_keywords:
    item['input'] = ''
    item['output'] = item['response']
    del item['response']

# save the filtered data to a json file

with open('airoboros_formatted.json', 'w') as outfile:
    json.dump(filtered_data_keywords, outfile, indent=1)