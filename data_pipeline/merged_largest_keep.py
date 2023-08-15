import json
import glob
import argparse
def load_and_filter(file_path):
    # Load the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Filter out entries with None values
    filtered_data = []
    for entry in data:
        if entry.get('instruction') is not None and entry.get('output') is not None:
            filtered_data.append(entry)
    return filtered_data

# Merge and deduplicate data
def merge_and_deduplicate(file_paths):
    merged_data = []
    duplicates_removed = 0  

    for file_path in file_paths:
        merged_data += load_and_filter(file_path.strip())

    # Deduplicate the merged data
    unique_data = {}
    for item in merged_data:
        instruction = item['instruction']
        input_data = item.get('input')  # Some entries may not have an 'input' field

        # Create a unique key using both 'instruction' and 'input'
        unique_key = f"{instruction}-{input_data}"

        if unique_key in unique_data:
            print('Duplicate entry:', unique_key)
            duplicates_removed += 1
            # Keep the one with the longer 'output' field
            if len(item['output']) > len(unique_data[unique_key]['output']):
                
                unique_data[unique_key] = item
        else:
            unique_data[unique_key] = item

    # Write the unique data back into a list
    unique_data_list = list(unique_data.values())

    print('Total duplicates removed:', duplicates_removed)
    
    return unique_data_list

# Set up the argument parser
parser = argparse.ArgumentParser(description='Merge and deduplicate JSON files.')
parser.add_argument('file_paths', metavar='file_path', type=str, nargs='*', help='the paths to the files you want to merge')

# Parse the arguments
args = parser.parse_args()

if args.file_paths:
    file_paths = args.file_paths
else:
    file_paths = glob.glob("*.json")

unique_data_list = merge_and_deduplicate(file_paths)
print('Total Examples:', len(unique_data_list))

with open('merged_and_deduped.json', 'w') as file:
    json.dump(unique_data_list, file, indent=1)