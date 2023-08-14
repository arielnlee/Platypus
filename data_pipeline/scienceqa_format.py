from datasets import load_dataset

import pandas as pd
dataset = load_dataset('metaeval/ScienceQA_text_only') 

# Load in to a df
df = pd.DataFrame(dataset['train'])
# Print out all of the unique first three words of the question
unique_first_three_words = df['question'].apply(lambda x: ' '.join(x.split()[:3])).unique()
print(unique_first_three_words)

# Print out all the skills
skills = df['skill'].unique().tolist()
for skill in sorted(skills):
    print(skill)
print(len(df['skill'].unique()))

skills_to_remove = ['Choose customary units of distance','Choose customary units of mass','Choose customary units of volume','Is it a complete sentence or a run-on','Is the sentence simple compound, comples, or compound-complex','Use guide words']
print('Length before removing skills: ', len(df))
df = df[~df['skill'].isin(skills_to_remove)]
print('Length after removing skills: ', len(df))


# remove the task, grade, subject,topic, category, and skill columns
df_dropped = df.drop(['task', 'grade', 'subject', 'topic', 'category', 'skill'], axis=1)
df_dropped.head(10)

# Check if every example has a choices field that is not an empty string
print('Length of updated df:',len(df_dropped))
df_dropped['solution'].apply(lambda x: x != '').value_counts()

# Filter the df to only include examples with a non-empty choices field

df_filtered = df_dropped[df_dropped['solution'].apply(lambda x: x != '')]
print('Keeping only those questions which have a long-form solution:',len(df_filtered))
# Reset the row numbers of df_filtered
df_filtered = df_filtered.reset_index(drop=True)

# Print out a couple of random examples in an easier to read format

import random
for i in range(5):
    random_index = random.randint(0, len(df_filtered)-1)
    print("Question: ", df_filtered['question'][random_index])
    print("Choices: ", df_filtered['choices'][random_index])
    print("Answer: ", df_filtered['answer'][random_index])
    print("Solution: " , df_filtered['solution'][random_index])
    print("Context: ", df_filtered['lecture'][random_index])
    print("\n")

# Create a new df with the columns we want to keep
df_reformatted = df_filtered[['question', 'choices', 'solution', 'lecture', 'answer']]

# Add a column which contains the correct answer based on the answer index for the list
df_reformatted['correct_answer'] = df_reformatted.apply(lambda x: x['choices'][x['answer']], axis=1)

# Reformat the choices column to be a string of the form: A, choice1, B, choice2, C, choice3, D, choice4
df_reformatted['choices'] = df_reformatted['choices'].apply(lambda x: '\n'.join([f'{chr(65+i)}: {choice}' for i, choice in enumerate(x)]))

# Combine the question and choices columns into one column
df_reformatted['question'] = df_reformatted['question'] + '\n' + df_reformatted['choices']

# Rename question to instruction, lecture to input, and solution to output
df_reformatted = df_reformatted.rename(columns={'question': 'instruction', 'lecture': 'input', 'solution': 'output','answer':'answer'})

# reorder the columns to instruction, input, output, correct_answer, answer
df_reformatted = df_reformatted[['instruction', 'input', 'output', 'correct_answer','answer']]

print('Length of df_reformatted:', len(df_reformatted))

# Remove the examples which have duplicate inputs
df_reformatted = df_reformatted.drop_duplicates(subset=['instruction'])
print('Length of df_reformatted after removing duplicates:', len(df_reformatted))

# Display a graph of the histogram of the lengths of the outputs based on words
import matplotlib.pyplot as plt

# Get the lengths of the outputs
output_lengths = df_reformatted['output'].apply(lambda x: len(x.split()))

# Plot the histogram
plt.hist(output_lengths, bins=30)
plt.xlabel('Length of output')
plt.ylabel('Number of examples')
plt.title('Histogram of lengths of outputs')
plt.show()

print('There are ',len(output_lengths[output_lengths > 40]), 'examples with output length > 40')

# Print the number of unique inputs
print('There are', len(df_reformatted['input'].unique()), 'unique inputs\n')

# Print the unique inputs
df_reformatted['input'].unique()

# For every input, print the number of examples with that input
for input in df_reformatted['input'].unique():
    print('There are', len(df_reformatted[df_reformatted['input'] == input]), 'examples with input:', input[:60])
    print('\n')

# print the first 5 examples from df_reformatted
for i in range(50):
    random_index = random.randint(0, len(df_reformatted)-1)
    print("Instruction:\n",df_reformatted['instruction'][random_index], "\n")
    print("Input:\n", df_reformatted['input'][random_index], "\n")
    print("Output:\n", df_reformatted['output'][random_index], "\n")
    print("Correct Answer:\n", df_reformatted['correct_answer'][random_index], "\n")
    print("\n")

# List of the questions that need description
need_description = ['What information supports','Based on','Read the following','Use the evidence','Look at the','According to a']

# Removing the input for those which don't need it
def check_description(instruction):
    for desc in need_description:
        if desc in instruction:
            return True
    return False

# Apply the function to the 'instruction' column
df_reformatted['contains_description'] = df_reformatted['instruction'].apply(check_description)

# Now update 'input' field where 'contains_description' is False
df_reformatted.loc[~df_reformatted['contains_description'], 'input'] = ''

# Plot of average input length vs output length 
input_lengths = df_reformatted['input'].apply(lambda x: len(x.split()))
output_lengths = df_reformatted['output'].apply(lambda x: len(x.split()))

plt.scatter(input_lengths, output_lengths)
plt.xlabel('Input length')

plt.ylabel('Output length')
plt.title('Input length vs output length')

plt.show()

# remove the examples with input length > 400
df_reformatted = df_reformatted[input_lengths <= 400].reset_index(drop=True)

# save the information from the df_reformatted to a json file, with the format: {"instruction": "instruction text", "input": "input text", "output": "output text"}

df_reformatted.drop(['correct_answer','answer','contains_description'], axis=1).to_json('scienceqa_formatted.json', orient='records',indent=1)