import pandas as pd
from datasets import load_dataset

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

# Loading the data
guanaco = load_dataset('timdettmers/openassistant-guanaco',split='train')
guanaco_test = load_dataset('timdettmers/openassistant-guanaco',split='test')

# Put the guanaco data in a  dataframe from the guanaco and guanaco_test variables
df = pd.DataFrame(guanaco)
df_test = pd.DataFrame(guanaco_test)
df = pd.concat([df, df_test])
df.head()

def split_text(text):
    split_marker = "### Assistant:"
    instruction, output = text.split(split_marker, 1)
    instruction = instruction.replace("### Human:", "").strip()
    output = output.replace("### Human:", "### Instruction:\n").replace("### Assistant:", "### Response:\n").strip()
    return pd.Series([instruction, output])

# Apply the function to the filtered dataframe
df[['instruction', 'output']] = df['text'].apply(split_text)

df.head()

# Add word boundary to the pattern to match exact words
pattern = r'\b(' + '|'.join(keywords_expanded) + r')\b'

# Filter the dataframe
df_filtered = df[df['instruction'].str.contains(pattern, case=False, na=False)]

print(len(df_filtered))

import json
data = df_filtered[['instruction', 'output']].to_dict('records')
for record in data:
    record['input'] = ''

data_reordered = []
for record in data:
    data_reordered.append({'instruction': record['instruction'], 'input': record['input'], 'output': record['output']})

with open('guanaco_formatted.json', 'w') as f:
    json.dump(data_reordered, f, indent=1)