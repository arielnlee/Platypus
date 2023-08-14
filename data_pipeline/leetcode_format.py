from datasets import load_dataset
import pandas as pd
dataset = load_dataset("nuprl/leetcode-solutions-python-testgen-gpt4")
df = pd.DataFrame(dataset['train'])

# Pull out the prompt and full_code columns

df = df[['prompt', 'full_code']]

# add a column named 'input' which is empty between those two

df['input'] = ''

# rename promt to instruction and full_code to output   

df = df.rename(columns={'prompt': 'instruction', 'full_code': 'output'})

# move output to the end

df = df[['instruction', 'input', 'output']]

# save to json

df.to_json('leetcode_formatted.json', orient='records',indent=1)