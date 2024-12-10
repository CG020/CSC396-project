import os
import pandas as pd
import chardet


KAGGLE_DIR = "data/train/kaggle_set"
MTS_DIALOG = "data/train/MTS-Dialog-TrainingSet.csv"
VA_GOV = "data/Transcripts/Transcripts"


def parse_transcripts(directory):
    conversations = []
    conversation_id = 0 
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            conversation_id += 1  
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                dialogue_accumulator = [] 
                speaker = None
                category = 'Unknown' 

                for line in file:
                    line = line.strip()
                    if "DOCTOR" in line:
                        if dialogue_accumulator and speaker:
                            conversations.append({
                                'conversation_id': conversation_id,
                                'category': category,
                                'speaker': speaker,
                                'text': ' '.join(dialogue_accumulator).strip()
                            })
                            dialogue_accumulator = [] 
                        speaker = 'doctor'
                    elif any(prefix in line for prefix in ["PATIENT", "SECOND PERSON", "THIRD PERSON"]):
                        if dialogue_accumulator and speaker:
                            conversations.append({
                                'conversation_id': conversation_id,
                                'category': category,
                                'speaker': speaker,
                                'text': ' '.join(dialogue_accumulator).strip()
                            })
                            dialogue_accumulator = [] 
                        speaker = 'patient'
                    else:
                        dialogue_accumulator.append(line)

                if dialogue_accumulator and speaker:
                    conversations.append({
                        'conversation_id': conversation_id,
                        'category': category,
                        'speaker': speaker,
                        'text': ' '.join(dialogue_accumulator).strip()
                    })

    return conversations


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
    # kaggle_conv = parse_kaggle(KAGGLE_DIR)
    # save_parsed(kaggle_conv, "data/parsed_kaggle.csv")
    
    # mts_dialog_conv = parse_mts_dialog(MTS_DIALOG)
    # save_parsed(mts_dialog_conv, "data/parsed_mts_dialog.csv")
    gov_conv = parse_transcripts(VA_GOV)
    save_parsed(gov_conv, "data/parsed_va_gov.csv")
    
    print("Parsing completed - check data folder")