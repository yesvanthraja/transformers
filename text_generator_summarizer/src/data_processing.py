import os
import csv
import pandas as pd


def process_chat_data(data_path):
    # import pdb;pdb.set_trace()
    try:
        if os.path.exists(data_path):
            # prompts = {}
            data = pd.read_csv(data_path, quoting=csv.QUOTE_ALL)
            input_prompt = data['input'].tolist()
            output_prompt = data['output'].tolist()
            # prompts["input_prompt"] = input_prompt
            # prompts["output_prompt"] = output_prompt
            return input_prompt, output_prompt
    except Exception as e:
        return {"Error": str(e)}


def process_summarizer_data(data_dir):
    summarized_text = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(filename, data_dir), "r", encoding = "utf-8") as f:
                lines = f.read()
                summarized_text.append(lines)
    return summarized_text


