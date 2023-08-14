from datasets import load_dataset
import pandas as pd
import json

dataset = load_dataset('openbookqa', name='additional')

# put data in a dataframe
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])
df_val = pd.DataFrame(dataset['validation'])

# Put the dataframes into a single dataframe
df = pd.concat([df_train, df_test, df_val])
df.head()

# Convert the choices into multiple choice format
df['choices'] = df['choices'].apply(lambda x: [x['text'][i] for i in range(len(x['text']))])

# Start with an empty list to hold all the new JSON objects
json_objects_updated = []

# For each row in the dataframe
for idx, row in df.iterrows():
    # Parse the choices string into a list
    choices = row['choices']
    
    # Format the choices with alphabetic indicators
    formatted_choices = '\n'.join([f'\n{chr(65+i)}: {choice}' if i == 0 else f'{chr(65+i)}: {choice}' for i, choice in enumerate(choices)])
    
    # Combine the question stem with the formatted choices
    instruction = row['question_stem'] + ': ' + formatted_choices
    
    # Get the correct answer based on the answer key
    correct_answer = choices[ord(row['answerKey']) - 65]
    
    # Format the output with a random answer prefix from the updated list, the correct answer key, and the correct answer
    output = 'The answer is ' + row['answerKey'] + ": " + correct_answer
    
    # Create the JSON object and append it to the list
    json_objects_updated.append({
        "instruction": instruction,
        "input": row['fact1'],
        "output": output
    })

# Create a JSON file with the updated JSON objects, with an indent of 1 for readability
with open('openbookqa_formatted.json', 'w') as f:
    json.dump(json_objects_updated, f, indent=1)