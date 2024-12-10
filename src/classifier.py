import torch
import torch.nn as nn
import torch.nn.functional as F

## CLASSIFIER ARCHITECTURE

# feed forward neural network + configurable layers for binary classification


# binary classififer + feed forward
class MedicalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]): # sizes define the numeber of nuerons each layer gets
        super().__init__()
        layers = []
        prev_dim = input_dim # size of the dimensions for what is inputted 
        
        for dim in hidden_dims:
            layers.extend([ # the added layers
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim), # normalization
                nn.Dropout(0.3), # regulatization
            ])
            prev_dim = dim
            
        self.shared = nn.Sequential(*layers)
        self.severity = nn.Linear(hidden_dims[-1], 1)
        self.solved = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x):
        # forwarding technique
        features = self.shared(x) # creates a features set from the layers in the classifier 
        # predictions for both layers
        severity_pred = self.severity(features)
        solved_pred = self.solved(features)
        
        # combines severe and solved 
        return torch.cat((severity_pred, solved_pred), dim=1)


class MedicalDataset(torch.utils.data.Dataset):
    # functioning for handling the labels of the dataset
    def __init__(self, x, y, vocab_size):
        if len(x) != len(y):
            raise ValueError(f"Feature length ({len(x)}) does not match label length ({len(y)})")
        self.x = x
        self.y = y
        self.vocab_size = vocab_size
        
    def __len__(self): # total sample number in dataset
        return len(self.x)
    
    def __getitem__(self, index): # iterates over the dataset 
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")
            
        # asserts labels onto the data in the size dimension
        x = torch.zeros(self.vocab_size, dtype=torch.float32)
        y = torch.tensor(self.y[index])
        
        for k,v in self.x[index].items():
            if not 0 <= k < self.vocab_size:
                raise ValueError(f"Token ID {k} out of vocabulary range [0, {self.vocab_size})")
            x[k] = v
            
        return x, y