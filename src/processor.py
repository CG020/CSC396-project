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
        doc_text = ' '.join(group[group['speaker'] == 'doctor']['text']).lower()
        pat_text = ' '.join(group[group['speaker'] == 'patient']['text']).lower()
        
        # removing the questions happens here

        # doc_text = remove_questions(doc_text)

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
            'pain',
            'severe',
            'worse',
            'chest',
            'stabbing',
            'aching',
            'hurting',
            'burning',
            'shortness',
            'breathing',
            'wheeze',
            'coughing',
            'wheezing',
            'fever',
            'vomiting',
            'nausea',
            'chills',
            'diarrhea',
            'flu',
            'congestion',
            'heart',
            'palpitations',
            'pressure',
            'attack',
            'dizziness',
            'confusion',
            'fainting',
            'tingling',
            'numbness',
            'cancer',
            'asthma',
            'diabetes',
            'infection',
            'pneumonia',
            'arthritis',
            'stroke',
            'disease',
            'failure',
            'hospitalized',
            'loss',
            'urgent',
            'emergency',
            'changes',
            'significant',
            'worsening',
            'swelling',
            'weakness',
            'fatigue',
            'swollen',
            'bleeding',
            'blood',
            'kidney',
            'lung'
        ]
        is_severe = any(term in text for term in severe_indicators)
        
        solved_indicators = [
            'ok',
            'better',
            'improved',
            'resolved',
            'fine',
            'alright',
            'nothing',
            'yeah',
            'good',
            'stable',
            'clear',
            'normal',
            'recovered',
            'gone',
            'healthy',
            'better',
            'well',
            'resting',
            'okay',
            'fine-tuned',
            'manageable',
            'controlled',
            'resolved',
            'healed',
            'cleared',
            'reassured'
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