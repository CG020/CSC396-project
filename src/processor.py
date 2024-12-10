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
    
    # conversation ids in the data are the indeces of separate conversations
    for conv_id, group in tqdm(df.groupby('conversation_id'), desc='Processing conversations'):
        # two parties in question speaker and doctor are their own columns
        doc_text = ' '.join(group[group['speaker'] == 'doctor']['text'].fillna('').astype(str)).lower()
        doc_words = [word.strip('"').strip('?').strip('.').strip(',') for word in doc_text.split()]
        doc_text = ' '.join(doc_words)

        pat_text = ' '.join(group[group['speaker'] == 'patient']['text'].fillna('').astype(str)).lower()
        pat_words = [word.strip('"').strip('?').strip('.').strip(',') for word in pat_text.split()]
        pat_text = ' '.join(pat_words)
        
        # combined string 
        full_text = f"<doctor> {doc_text} </doctor> <patient> {pat_text} </patient>"
        processed_data.append({
            'conversation_id': conv_id,
            'text': full_text,
            'category': group['category'].iloc[0]
        })
    
    return pd.DataFrame(processed_data)


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


def create_vector(tokens, token_to_id, unk_id=0):
    # vector creation
    vector = defaultdict(int)
    for t in tokens:
        i = token_to_id.get(t, unk_id)
        vector[i] += 1
    return vector


def create_labels(df):
    labels = []
    
    # some prelimary indicators for the binary classification
    for _, row in tqdm(df.iterrows(), desc='Creating labels', total=len(df)):
        text = row['text'].lower()
        
        severe_indicators = [
            'ache',
            'aches',
            'aching',
            'arthritis',
            'asthma',
            'attack',
            'bled',
            'bleed',
            'bleeding',
            'blood',
            'breathing',
            'breathlessness',
            'burn',
            'burned',
            'burning',
            'cancer',
            'changes',
            'chest',
            'chills',
            'confusion',
            'congested',
            'congestion',
            'coughing',
            'diabetes',
            'diarrhea',
            'disease',
            'dizzier',
            'dizziness',
            'dizzy',
            'emergency',
            'fail',
            'failed',
            'fails',
            'failure',
            'faint',
            'fainted',
            'fainting',
            'fatigue',
            'fever',
            'feverish',
            'flu',
            'heart',
            'hospitalized',
            'hurt',
            'hurting',
            'hurts',
            'infection',
            'injury',
            'irritation',
            'irritations',
            'kidney',
            'kidneys',
            'loss',
            'lung',
            'lungs',
            'nausea',
            'numbness',
            'pain',
            'painful',
            'pains',
            'palpitations',
            'pneumonia',
            'pressing',
            'pressure',
            'rash',
            'severe',
            'short',
            'shorter',
            'shortness',
            'significant',
            'stabbing',
            'stiff',
            'stiffer',
            'stiffness',
            'stroke',
            'swell',
            'swelling',
            'swells',
            'swollen',
            'tingling',
            'trauma',
            'urgent',
            'urgently',
            'vomiting',
            'weak',
            'weakened',
            'weakness',
            'wheeze',
            'wheezing',
            'worse',
            'worsened',
            'worsening'
        ]
        is_severe = any(term in text for term in severe_indicators)
        
        solved_indicators = [
            'alright',
            'better',
            'clear',
            'cleared',
            'controlled',
            'diagnose',
            'diagnosed',
            'diagnoses',
            'diagnosing',
            'effective',
            'fine',
            'fine-tuned',
            'gone',
            'good',
            'heal',
            'healed',
            'healing',
            'heals',
            'healthy',
            'improve',
            'improved',
            'improvement',
            'improves',
            'improving',
            'manageable',
            'normal',
            'nothing',
            'ok',
            'okay',
            'prescribe',
            'prescribed',
            'prescribes',
            'prescribing',
            'reassured',
            'recovered',
            'resolved',
            'rest',
            'rested',
            'resting',
            'rests',
            'stable',
            'therapy',
            'treat',
            'treated',
            'treating',
            'treatment',
            'treatments',
            'treats',
            'well',
            'yeah'
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
    
    return train_features, dev_features, train_labels, dev_labels, vocab_size, train_df, dev_df