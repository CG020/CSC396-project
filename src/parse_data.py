import os
import pandas as pd
import chardet

KAGGLE_DIR = "data/train/kaggle_set"
MTS_DIALOG = "data/train/MTS-Dialog-TrainingSet.csv"


def parse_kaggle(directory):
    conversations = []
    conversation_id = 0 
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            conversation_id += 1
            category = filename[:3]
            
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as raw_file:
                result = chardet.detect(raw_file.read())
                encoding = result['encoding']

            with open(file_path, 'r', encoding=encoding) as file:
                for line in file:
                    if line.startswith("D: "):
                        conversations.append({
                            'conversation_id': conversation_id,
                            'category': category, 
                            'speaker': 'doctor',
                            'text': line[3:].strip()
                        })
                    elif line.startswith("P: "):
                        conversations.append({
                            'conversation_id': conversation_id,
                            'category': category, 
                            'speaker': 'patient',
                            'text': line[3:].strip()
                        })
    return conversations


def parse_mts_dialog(filepath):
    df = pd.read_csv(filepath)
    conversations = []
    
    for conversation_id, row in df.iterrows():
        category = row['section_header']
        dialogue = row['dialogue']
        
        for line in dialogue.splitlines():
            line = line.strip()
            
            if line.startswith("Doctor:"):
                conversations.append({
                    'conversation_id': conversation_id + 1,
                    'category': category,
                    'speaker': 'doctor',
                    'text': line[8:].strip()
                })
            elif line.startswith("Patient:"):
                conversations.append({
                    'conversation_id': conversation_id + 1,
                    'category': category,
                    'speaker': 'patient',
                    'text': line[8:].strip()
                })
    
    return conversations


def save_parsed(data, output_file):
    pd.DataFrame(data).to_csv(output_file, index=False)


if __name__ == "__main__":
    kaggle_conv = parse_kaggle(KAGGLE_DIR)
    save_parsed(kaggle_conv, "data/parsed_kaggle.csv")
    
    mts_dialog_conv = parse_mts_dialog(MTS_DIALOG)
    save_parsed(mts_dialog_conv, "data/parsed_mts_dialog.csv")
    
    print("Parsing completed - check data folder")
