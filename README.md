# CSC396-project

## Proposal for Project

Project Idea: We plan to make a neural network classifier focused on taking doctor-patient conversations (transcripts essentially), connect the conversation to a database of medical diagnoses that aren't focused necessarily on complex medical language since it is unlikely a doctor will use the scientific names in a patient consultation, but more common diagnoses gathered from a list of symptoms that either the patient or the doctor can identify. The classifier will analyze a conversation, it can classify the severity of the patient's problem based on symptoms and (possible) diagnosis and can classify how 'solved' the case was maybe by determining if a conclusive diagnosis and treatment was actually reached in conversation or not. Taking the binary classification approach this can translate to isSevere and isSolved.

## Calrification for Verification

isSevere: For isSevere evaluation there is database that weighs disease/illness symptoms based on how long they last that we could train on, if a high enough threshold is reached, a number we haven't determined (maybe lasting more than a week in the body) isSevere would be checked for the conversation we are analyzing.

isSolved: Connecting to a dataset where treamments are listed (be they in form of medicine or procedures) if they are present in the conversation and discussed at length with words that hint to a treatment plan being involved like 'pharmacy' or 'hospital', isSolved will be marked.

Surdeanu suggested:  recruit one student from the medical school to check the system outputs

## Dataset Possibilities
https://github.com/abachaa/MTS-Dialog/blob/main/Augmented-Data/

MTS-Dialog-Augmented-TrainingSet-1-En-FR-EN-2402-Pairs.csv
https://www.kaggle.com/datasets/azmayensabil/doctor-patient-conversation-large?select=MSK0008.txt
