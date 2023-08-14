#! /usr/bin/env python3
import json
import glob
import re
import argparse
import os
import subprocess
# Script to update the MATH dataset with enhanced solutions from the PRM dataset
# See convert_prm.py for the creating the required PRM file

def download_files():
    url = 'https://people.eecs.berkeley.edu/~hendrycks/MATH.tar'
    tar_file = 'MATH.tar'
    output_dir = 'MATH'

    # Check if the directory already exists
    if not os.path.isdir(output_dir):
        # If the directory doesn't exist, download the tar file
        try:
            subprocess.check_output(["curl", "-L", url, "-o", tar_file])
        except subprocess.CalledProcessError as e:
            print("Curl failed with error:", e.output)
            return

        # Try to unzip
        try:
            print("Unzipping", tar_file, "...")
            subprocess.check_output(["tar", "-xf", tar_file])
        except subprocess.CalledProcessError as e:
            print("Tar failed with error:", e.output)
            return

        # Remove the tar file
        try:
            subprocess.check_output(["rm", tar_file])
        except subprocess.CalledProcessError as e:
            print("Remove tar file failed with error:", e.output)
            return
    else:
        print("Directory", output_dir, "already exists, skipping download")

    
def main(args):
    download_files()
    # read in the .jsonl file
    with open(args.prm_file, 'r') as f:
        lines = f.readlines()
    jsonl_data = [json.loads(line) for line in lines]

    def process_directory(directory):
        # find all .json files in all subdirectories
        filepaths = glob.glob(directory + '/**/*.json', recursive=True)
        
        combined_data = []
        for filepath in filepaths:
            with open(filepath, 'r') as f:
                data = json.load(f)

                # Replace 'problem' with 'input' and 'solution' with 'output'
                modified_data = {
                    'instruction': data['problem'],
                    'input': "",
                    'output': data['solution'],
                }

                combined_data.append(modified_data)

        # save the combined data to a .json file
        dir_name = directory.split('/')[-1]
        with open(f'MATH_{dir_name}_data.json', 'w') as f:
            json.dump(combined_data, f)

        return combined_data


    # create single json files for train and test data
    train_data = process_directory('MATH/train')
    test_data = process_directory('MATH/test')

    # add train data and test data together to get all data
    train_data = train_data + test_data
    
    # train_data_locations = [data['instruction'] for data in train_data]

    # Put the train and test data together
    train_data_locations = [data['instruction'] for data in train_data]
    test_data_locations = [data['instruction'] for data in test_data]

    train_data_locations.extend(test_data_locations)

    count = 0
    # replace with the enhanced solutions
    for data in jsonl_data:
        if data['Input'] in train_data_locations:
            index = train_data_locations.index(data['Input'])

            answer = data['Output'].split('# Answer\n\n')[1]
            modified_content = {
                'instruction': data['Input'],
                'input': "",
                # 'output': re.sub(r'# Answer\n\n.*', r'\\boxed{' + re.escape(answer) + '}', data['Output']),
                'output': re.sub(r'# Answer\n\n.*','', data['Output']),
                # 'gt': answer
            }
            train_data[index] = modified_content
            count += 1

    # For every problem in the train_data, if the ouput has a boxed answer, remove the box update to keep the answer
    for data in train_data:
        if re.search(r'\\boxed{(.*)\}', data['output']):
            data['output'] = re.sub(r'\\boxed\{(.*)\}', r'\1', data['output'])

    # overwrite the train_data.json file
    with open('MATH_train_enhanced_no_boxed.json', 'w') as f:
        json.dump(train_data, f)

    print('Updated ' + str(count) + ' problems with enhanced solutions')

    # delete the intermediate files
    if args.delete_intermediate_files:
        #remove the math folder
        os.remove('MATH_test_data.json')
        os.remove('MATH_train_data.json')
        subprocess.check_output(["rm", "-rf", "MATH"])
        os.remove(args.prm_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prm_file', nargs='?',help='location of the PRM file (defaults to current directory if only filename is given))', default = 'prm_updated.json'),
    parser.add_argument('delete_intermediate_files', nargs='?',help='Delete intermediate files', default=True)
    args = parser.parse_args()
    main(args)


