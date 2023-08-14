from datasets import load_dataset

dataset = load_dataset("TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k", split="train")

# filter based on if the instruction says "Use pyton"

dataset = dataset.filter(lambda example: "python" in example["instruction"])  

# make df from dataset
import pandas as pd
df = dataset.to_pandas()

# swap the values of instruction and input
df['instruction'], df['input'] = df['input'], df['instruction']

# make input empty
df['input'] = ''

# save the df to json
df.to_json("tigerbot-kaggle-formatted.json", orient="records",indent=1)