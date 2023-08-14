import json
import requests
import pandas as pd

def fetch_data_from_url(url):
    headers = {'accept': 'application/json'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        return response.json()
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None
    
# MATH

url = 'https://arb.duckai.org/api/lib/math'
data = fetch_data_from_url(url)

if data is not None:
    # Extract the ids
    ids = [item['_id'] for item in data]

    formatted_data = []

    # For each id, make a request to the API and format the returned data
    for id_ in ids:
        response = requests.get(f"https://arb.duckai.org/api/lib/math/{id_}")
        response_data = response.json()

        # Format the data
        formatted_entry = {
            "instruction": response_data["Problem_Statement"],
            "input": "",
            "output": response_data["Solution"]
        }
        formatted_data.append(formatted_entry)

    # Create a DataFrame from the formatted data
    df = pd.DataFrame(formatted_data)

    # Save the DataFrame to a new JSON file
    df.to_json("formatted_math_data.json", orient="records", indent=1)
else:
    print("No data was returned from the API for Math")


#MCAT READING

url = 'https://arb.duckai.org/api/lib/mcatReading'
data = fetch_data_from_url(url)

if data is not None:
    # Create an empty list to hold the formatted data
    formatted_data = []

    # Loop through each item in the data
    for item in data:
        # Extract the instruction, possible solutions, and correct answer
        instruction = item['Problem Statement']
        options = item['Answer Candidates']
        output = item['Solution']

        output = output.split('.', 1)[-1].lstrip()

    # Append the options to the instruction
        for i, option in enumerate(options, start=65): 
            instruction += f"\n{chr(i)}. {option}"


        # Create a new dictionary with 'input' and 'output' switched
        formatted_entry = {
            "instruction": instruction,
            "input": "Choose A, B, C or D as your solution.",
            "output": output
        }

        # Add the formatted entry to the list
        formatted_data.append(formatted_entry)

    # Create a DataFrame from the formatted data
    df = pd.DataFrame(formatted_data)

    # Save the DataFrame to a new JSON file
    df.to_json("formatted_mcat_data.json", orient="records",indent=1)
else:
    print("No data was returned from the API for MCAT Reading")

# LAW

# Load the data from the file
url = 'https://arb.duckai.org/api/lib/law'
data = fetch_data_from_url(url)

if data is not None:
    # Create an empty list to hold the formatted data
    formatted_data = []

    # Loop through each item in the data
    for item in data:
        # Extract the instruction, possible solutions, and correct answer
        instruction = item['Problem Statement']
        options = item['Answer Candidates']
        output = item['Final Answer']

    # Append the options to the instruction
        for i, option in enumerate(options, start=65):  
            instruction += f"\n{chr(i)}. {option}"


        # Create a new dictionary with 'input' and 'output' switched
        formatted_entry = {
            "instruction": instruction,
            "input": "Choose A, B, C or D as your solution.",
            "output": output
        }

        # Add the formatted entry to the list
        formatted_data.append(formatted_entry)

    # Create a DataFrame from the formatted data
    df = pd.DataFrame(formatted_data)

    # Save the DataFrame to a new JSON file
    df.to_json("formatted_law_data.json", orient="records",indent=1)
else:
    print("No data was returned from the API for MCAT Reading")


#MCAT SCIENCE

url = 'https://arb.duckai.org/api/lib/mcatscience/val'
data = fetch_data_from_url(url)

if data is not None:
    # Create an empty list to hold the formatted data
    formatted_data = []

    # Loop through each item in the data
    for item in data:
        # Extract the instruction, possible solutions, and correct answer
        instruction = item['Problem Statement']
        options = item['Answer Candidates']
        output = item['Solution']

        output = output.split('.', 1)[-1].lstrip()

    # Append the options to the instruction
        for i, option in enumerate(options, start=65):  # ASCII value of 'A' is 65
            instruction += f"\n{chr(i)}. {option}"


        # Create a new dictionary with 'input' and 'output' switched
        formatted_entry = {
            "instruction": instruction,
            "input": "Choose A, B, C or D as your solution.",
            "output": output
        }

        # Add the formatted entry to the list
        formatted_data.append(formatted_entry)

    # Create a DataFrame from the formatted data
    df = pd.DataFrame(formatted_data)

    # Save the DataFrame to a new JSON file
    df.to_json("formatted_mcat_science_data.json", orient="records",indent=1)
else:
    print("No data was returned from the API for MCAT Science")

# PHYSICS

url = 'https://arb.duckai.org/api/lib/physics/val'
data = fetch_data_from_url(url)


if data is not None:
    # Extract the ids
    ids = [item['_id'] for item in data]

    formatted_data = []

    # For each id, make a request to the API and format the returned data
    for id_ in ids:
        try:
            response = requests.get(f"https://arb.duckai.org/api/lib/physics/val/{id_}")
            response_data = response.json()
            # Format the data
            formatted_entry = {
                "instruction": response_data["Problem_Statement"],
                "input": "",
                "output": response_data["Solution"]
            }
            formatted_data.append(formatted_entry)

        except:
            print(f"Error with id: {id_}")


    # Create a DataFrame from the formatted data
    df = pd.DataFrame(formatted_data)

    # Save the DataFrame to a new JSON file
    df.to_json("formatted_physics_data.json", orient="records", indent=1)
else:
    print("No data was returned from the API for Physics.")
