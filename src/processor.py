import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

tqdm.pandas()


# processing the medical datasets for training:
# waht is done --> tokenization, vocabulary creation, feature vector generation, and label determination based on specific keywords in the text (these can be expanded upon)
# data parsed for model format


def process_conversations(df):
    processed_data = []
    
    for conv_id, group in tqdm(df.groupby('conversation_id'), desc='Processing conversations'):
        doc_text = ' '.join(group[group['speaker'] == 'doctor']['text'])
        pat_text = ' '.join(group[group['speaker'] == 'patient']['text'])
        
        # removing the questions happens here

        doc_text = remove_questions(doc_text)

        # combined texts
        full_text = f"<doctor> {doc_text} </doctor> <patient> {pat_text} </patient>"
        processed_data.append({
            'conversation_id': conv_id,
            'text': full_text,
            'category': group['category'].iloc[0]
        })
    
    return pd.DataFrame(processed_data)

def remove_questions(text):
    sentences = text.split('.')
    filtered_sentences = [sentence.strip() for sentence in sentences if not sentence.strip().endswith('?')]
    return ' '.join(filtered_sentences)


def create_vocabulary(df, threshold=10):
    try:
        df['tokens'] = df['text'].progress_apply(word_tokenize)
    except Exception:
        import nltk
        nltk.download('punkt_tab')
        nltk.download('punkt')
        df['tokens'] = df['text'].progress_apply(word_tokenize)
    tokens = df['tokens'].explode().value_counts()
    tokens = tokens[tokens > threshold] # threshold is the limit to reach for a word to be included in the vocab
    
    id_to_token = ['[UNK]'] + tokens.index.tolist()
    token_to_id = {w:i for i,w in enumerate(id_to_token)}
    
    return token_to_id, len(id_to_token)

def create_vector(tokens, token_to_id, max_length=500):
    vector = [token_to_id.get(token, 0) for token in tokens[:max_length]]
    if len(vector) < max_length:
        vector += [0] * (max_length - len(vector))
    return vector


def create_labels(df):
    labels = []
    
    # some prelimary indicators for the binary classification
    for _, row in tqdm(df.iterrows(), desc='Creating labels', total=len(df)):
        text = row['text'].lower()
        
        severe_indicators = severe_indicators = [
            'severe', 'emergency', 'worst', 'extreme', 'fever',
            'chest pain', 'shortness of breath', 'high blood pressure', 
            'low blood pressure', 'rapid heart rate', 'difficulty breathing', 
            'unconscious', 'fainting', 'excruciating', 'unbearable', 
            'intense pain', 'severe pain', 'immediate attention', 
            'urgent care', 'emergency room', 'sepsis', 'heart attack', 
            'stroke', 'internal bleeding', 'critical condition', 
            'rapidly worsening', 'severe complications', 'requires immediate',
            'unstable', 'life-threatening', 'acute distress',
            'emergency intervention', 'critically ill', 'severe reaction', 
            'acute onset', 'collapse', 'critically low', 'dangerously high', 
            'irregular heartbeat', 'rapid pulse', 'shallow breathing', 
            'labored breathing', 'respiratory distress', 'blood pressure crisis',
            'agonizing pain', 'debilitating pain', 'crippling pain',
            'incapacitating', 'intolerable', 'crushing chest', 'shooting pain', 
            'hemorrhage', 'seizure', 'convulsions', 'anaphylaxis',
            'cardiac arrest', 'heart failure', 'pulmonary embolism',
            'severe infection', 'meningitis', 'septic shock',
            'loss of consciousness', 'unresponsive', 'disoriented',
            'severe confusion', 'delirium', 'severe anxiety attack',
            'severe injury', 'major trauma', 'head injury',
            'massive bleeding', 'severe burn', 'fracture',
            'code blue', 'intensive care', 'critical care needed',
            'immediate surgery', 'stat', 'level one trauma',
            'acute abdomen', 'severe dehydration'
        ]
        is_severe = any(term in text for term in severe_indicators)
        
        solved_indicators = [
            'prescription', 'treatment', 'resolved', 'better',
            'follow up', 'thank you', 'prescribed', 'medication', 
            'dosage', 'pills', 'tablets', 'treatment plan', 
            'care plan', 'management plan', 'improved', 'feeling better', 
            'symptoms resolved', 'recovery', 'follow-up appointment', 
            'check back', 'monitor', 'discharge', 'can go home', 
            'cleared to leave', 'symptoms alleviated', 'condition stable',
            'marked improvement', 'normal range', 'successfully treated', 
            'response to treatment', 'continuing to improve', 
            'maintenance therapy', 'regular monitoring', 'preventive measures',
            'rehabilitation plan', 'follow-up care', 'remission', 
            'controlled', 'stabilized', 'managed', 'optimized', 
            'maintained', 'normalized', 'rehabilitated', 'symptoms subsided', 
            'pain free', 'infection cleared', 'test negative', 'normal levels', 
            'within normal limits', 'healing well', 'good progress',
            'surgery successful', 'procedure completed',
            'treatment course completed', 'therapy completed',
            'final dose', 'final session', 'treatment concluded',
            'routine checkup', 'maintenance dose', 'preventative care',
            'fully recovered', 'back to normal', 'resumed activities',
            'pain controlled', 'symptoms managed', 'condition resolved',
            'good prognosis', 'satisfactory progress', 'clinically stable',
            'no complications', 'no concerns', 'no further treatment',
            'discharged home', 'release papers', 'clearance given',
            'treatment summary', 'case closed', 'final visit',
            'asymptomatic', 'pain-free', 'fully functional',
            'return to work', 'return to activities', 'self-managing'
        ]
        is_solved = any(term in text for term in solved_indicators)
        
        labels.append([int(is_severe), int(is_solved)])
    
    return np.array(labels)


# preparation of data for classifier training
def prepare_data(df, test_size=0.2):
    print("Processing conversations...")
    processed_df = process_conversations(df)
    
    print("Creating vocabulary...")
    token_to_id, vocab_size = create_vocabulary(processed_df)
    
    print("Creating features...")
    processed_df['features'] = processed_df['tokens'].progress_apply(
        lambda x: create_vector(x, token_to_id))
    
    print("Creating labels...")
    labels = create_labels(processed_df)
    
    print("Splitting data...")
    # training sets + development sets (for validation)
    train_df, dev_df = train_test_split(processed_df, train_size=1-test_size)
    train_labels, dev_labels = train_test_split(labels, train_size=1-test_size)
    
    train_features = train_df['features'].tolist()
    dev_features = dev_df['features'].tolist()
    
    return train_features, dev_features, train_labels, dev_labels, vocab_size