import torch
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import os
from torch.utils.data import DataLoader 

from classifier import MedicalClassifier, MedicalDataset
from processor import prepare_data
from trainer import train_model

def main():
    # gpu setup
    use_gpu = True
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f'device: {device.type}')
    
    seed = 1234
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kaggle_path = os.path.join(project_root, 'data', 'train', 'parsed_kaggle.csv')
    mts_path = os.path.join(project_root, 'data', 'train', 'parsed_mts_dialog.csv')

    df = pd.concat([
        pd.read_csv(kaggle_path),
        pd.read_csv(mts_path)
    ])
    
    # data processesing for trainoing preparation
    train_features, dev_features, train_labels, dev_labels, vocab_size = prepare_data(df)
    
    # creates the datasets for training and validation
    train_ds = MedicalDataset(train_features, train_labels, vocab_size)
    dev_ds = MedicalDataset(dev_features, dev_labels, vocab_size)
    
    # batch processesing 
    batch_size = 32
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size)
    
    # create the model
    model = MedicalClassifier(input_dim=vocab_size).to(device)
    
    # training model
    train_loss, dev_loss = train_model(model, train_dl, dev_dl, device, n_epochs=10)
    
    # save the model for repeated use
    torch.save(model.state_dict(), 'medical_classifier.pt')

if __name__ == "__main__":
    main()