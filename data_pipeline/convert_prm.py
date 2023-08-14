#!/usr/bin/env python3

import json
import random
import argparse
import subprocess
import os

# Change this is you also want the ones were the final solution wasn't found
ONLY_SOLVED = True  
ONLY_FOUND_ANSWER = True

# Script to convert the openai prm800K dataset into input output pairs 

def download_files():
    urls = ["https://github.com/openai/prm800k/raw/main/prm800k/data/phase1_test.jsonl",
            "https://github.com/openai/prm800k/raw/main/prm800k/data/phase1_train.jsonl",
            "https://github.com/openai/prm800k/raw/main/prm800k/data/phase2_test.jsonl",
            "https://github.com/openai/prm800k/raw/main/prm800k/data/phase2_train.jsonl"]

    output_file_names = [name.split('/')[-1] for name in urls]
    for i in range(len(urls)):
        url = urls[i]
        output_file = output_file_names[i]

        # Check if the file already exists
        if not os.path.isfile(output_file):
            # If the file doesn't exist, download it
            try:
                subprocess.run(["curl", "-L", url, "-o", output_file])
                # print("Downloaded", output_file, "successfully!")
            except:
                # If the curl command fails, try wget
                subprocess.run(["wget", url, "-O", output_file])
                # print("Downloaded", output_file, "successfully!")
        else:
            print("File", output_file, "already exists, skipping download")

    return output_file_names 


def convert_format(data, ONLY_SOLVED):
    # Check for solved questions
    if ONLY_SOLVED and data["label"]["finish_reason"] != "solution":
        return None

    input = data["question"]["problem"]
    answer = data["question"]["ground_truth_answer"]

    steps = data["label"]["steps"]
    output = []
    answer_marker = "# Answer\n\n"
    for step in steps:
        # Get a set of correct completions
        completions = step.get("completions")
        selected_text = None
        if completions is not None:
            rated_completions = [comp for comp in completions if comp["rating"] == 1]
            completions_with_answer = [comp for comp in rated_completions if answer_marker in comp["text"]]
            if completions_with_answer:
                selected_completion = random.choice(completions_with_answer)
                selected_text = selected_completion["text"]
            elif rated_completions:
                selected_completion = random.choice(rated_completions)
                selected_text = selected_completion["text"]
            
        if selected_text:
            output.append(selected_text)

        human_completion = step.get("human_completion")
        if human_completion and human_completion["rating"] == 1:
            output.append(human_completion["text"])

    full_output = " ".join(output)

    # Find answer in the output if available
    answer_start_index = full_output.rfind(answer_marker)
    if answer_start_index != -1:
        answer_end_index = answer_start_index + len(answer_marker)
        found_answer = full_output[answer_end_index:].strip()
    else:
        found_answer = None 
    return {
        "Input": input,
        "Output": full_output,
        "Answer": answer,
        "Found_Answer": found_answer,
    }


def main(args):
    output_file = args.output_file
    input_files = download_files()

    with open(output_file, 'w') as out_f:
        for input_file_name in input_files:
            with open(input_file_name, 'r') as input_file:  
                line_count = 0
                for line in input_file:  
                    line_count += 1
                    data = json.loads(line)
                    converted_data = convert_format(data, ONLY_SOLVED)
                    if converted_data is not None:
                        if ONLY_FOUND_ANSWER and converted_data["Found_Answer"] == None:
                            pass
                        elif converted_data["Found_Answer"] == converted_data["Answer"]:
                            out_f.write(json.dumps(converted_data) + '\n')
                # print the number of lines in the input file, including the file name
                print("Number of lines in", input_file_name, ":", line_count)  
    # Print the number of lines in the output file
    print("Number of lines in", output_file, ":", sum(1 for line in open(output_file)))

    if args.delete_intermediate_files:
        for input_file_name in input_files:
            os.remove(input_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", nargs='?',help="Name of output file, will be overwritten if it exists", default="prm_updated.json")
    parser.add_argument("delete_intermediate_files", nargs='?',help="Delete intermediate files", default=True)
    args = parser.parse_args()

    main(args)
