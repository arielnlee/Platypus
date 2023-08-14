import os
import json
import subprocess
import os
# The directory where the json files are stored
# dir_path = 'original'
output_file = 'scibench_formatted.json'

import subprocess


# Define the repository URL
repo_url = "https://github.com/mandyyyyii/scibench.git"
repo_name = "scibench"
target_subdir = "dataset/original"

# Clone the repository
result_clone = subprocess.run(["git", "clone", repo_url])

# Check if the clone command was successful
if result_clone.returncode == 0:
    print("Repository cloned successfully!")

    # Construct the path to the target directory within the cloned repository
    target_path = os.path.join(os.getcwd(), repo_name, target_subdir)
    print("Path to the target directory:", target_path)
else:
    print("An error occurred while cloning the repository.")

json_files = [f for f in os.listdir(target_path) if f.endswith('.json')]


new_data = []

# Iterate over all the files
for json_file in json_files:
    file_path = os.path.join(target_path, json_file)

    # Open each json file
    with open(file_path, 'r') as f:
        # Load the data
        file_data = json.load(f)
        
        # Transform the data
        for d in file_data:
            output = d.get('solution')
            if not output:
                output = d.get('answer_number')
            transformed_data = {
                "instruction": d.get('problem_text'),
                "input": '',
                "output": output
            }
            new_data.append(transformed_data)

# Save the transformed data to a new json file
with open(output_file, 'w') as f:
    json.dump(new_data, f, indent=1)

# Remove the cloned repository
result_remove = subprocess.run(["rm", "-rf", repo_name])

