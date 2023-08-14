import pandas as pd
import json
from datasets import load_dataset
# The path where the csv file is stored
file_path = 'test.csv'
output_file = 'theoremqa_formatted.json'
dataset = load_dataset("wenhu/TheoremQA", data_files="test.csv")
new_data = []

# Open the csv file
df = pd.DataFrame(dataset['train'])

# Transform the data
for _, row in df.iterrows():
    instruction = "{}\nRelevant Theorem: {}".format(row['Question'], row['theorem_def'])
    transformed_data = {
        "instruction": instruction,
        "input": '',
        "output": row['Answer']
    }
    new_data.append(transformed_data)

# Save the transformed data to a new json file
with open(output_file, 'w') as f:
    json.dump(new_data, f, indent=1)