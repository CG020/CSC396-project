import torch
import torch.nn as nn
import torch.nn.functional as F

## CLASSIFIER ARCHITECTURE

# feed forward neural network + configurable layers for binary classification


# binary classififer + feed forward
class MedicalClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1):
        super(MedicalClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # the RNN structure using LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # two output layers for classifications
        self.severity = nn.Linear(hidden_dim, 1)
        self.solved = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        last_hidden = hidden[-1]

        severity_pred = self.severity(last_hidden)
        solved_pred = self.solved(last_hidden)
        
        return torch.cat((severity_pred, solved_pred), dim=1)

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # no lomnger needs to account for vocab size
        
    def __len__(self):
        return len(self.x)
    
    # one tensor creeation for each iteration
    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.long), torch.tensor(self.y[index], dtype=torch.float32)