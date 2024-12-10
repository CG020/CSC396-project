# Explanation of datasets

## Dataset 1 - MST Git
https://github.com/abachaa/MTS-Dialog/blob/main/Augmented-Data/MTS-Dialog-Augmented-TrainingSet-1-En-FR-EN-2402-Pairs.csv

This is a medical dataset also used to train a model more for summarization purpose. 
Main-Dataset Folder is the raw data - untouched by editors for paraphrasing or cleaning so this is the set we should use before performing any text manipulations.

The CSV for training os MTS-Dialog-TrainingSet.csv - the others were th evaluations and validations for a separate project. 

In this csv, the text of the conversation between the doctor and the patient is held, over 1000 conversations. There are also clinical notes in section text which I believe are categories of topic that the conversation is one we can ignore.


## Dataset 2 - Kaggle

https://www.kaggle.com/datasets/azmayensabil/doctor-patient-conversation-large?select=MSK0008.txt

This is a kaggle dataset for doctor patient conversations, simply the raw conversation formatted in turns - each txt file is its own conversation.

Explanation from the page: *There are 272 mp3 audio files and 272 corresponding transcript text files. Each file is titled with three characters and four digits. RES stands for respiratory, GAS represents gastrointestinal, CAR is cardiovascular, MSK is musculoskeletal, DER is dermatological, and the four following digits represent the case number of the respective disease category.*


## Dataset 3 - US Department of Veteran Affairs

https://www.data.va.gov/dataset/Physician-patient-transcripts-with-4C-coding-analy/4qbs-wgct/data?no_mobile=true

Government dataset with 405 conversations between a doctor and a patient with the occasional 'SECOND' or 'THIRD' person chiming in with input which we stored under 'patient' dialog. These coversations were conducted at the Veterans Health Administration (VHA) medical center primary care clinics. "@@@" indicates the audio couldn't be transcribed at the time of recording - we ignored these entries in code.


## Details

Combined we would have 1,878 conversations to use - each dataset would need its own parsing details attributed to their formats. Not necessary to use all but can use samples to train our first models.

Both datasets 1 and 2 do come with their own medical categories if we want to incorporate that into our project - a venture for later, otherwise they are turn-based conversations between two people - labelled as such. 

Update:
The third dataset we added does not come categorized so we did not incoporate medical categorization into this project for sake of time and extra category assignment