
# CSC396-project

## Proposal for Project

Project Idea: We plan to make a neural network classifier focused on taking doctor-patient conversations (transcripts essentially), connect the conversation to a database of medical diagnoses that aren't focused necessarily on complex medical language since it is unlikely a doctor will use the scientific names in a patient consultation, but more common diagnoses gathered from a list of symptoms that either the patient or the doctor can identify. The classifier will analyze a conversation, it can classify the severity of the patient's problem based on symptoms and (possible) diagnosis and can classify how 'solved' the case was maybe by determining if a conclusive diagnosis and treatment was actually reached in conversation or not. Taking the binary classification approach this can translate to isSevere and isSolved.

## Calrification for Verification

isSevere: For isSevere evaluation, we check the conversation against a variety of disease symptoms that require hospitalization or urgent care. 

isSolved: For isSolved evaluation, we check the conversation against the identifiers we used for deteriminants of treatment reached, dischargement, or signs of recovery.

Surdeanu suggested:  recruit one student from the medical school to check the system outputs

## Dataset Used
Details on datasets in the README located in data foldern(data/README)

## Instructions on How to Run Code:
- requirements.txt should have the required libraries
- nltk download
- run python src/main.py or in src folder run main.py
- output will display the evaluation results and results are outputted to src/output.txt