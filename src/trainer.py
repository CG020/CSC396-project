import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm


# training script + quick evaluation
# handles weights and imbalances - sets up the optimizer

# this is where parameter changes can occur !!

def calculate_class_weights(labels):
    # weighting function
    labels = labels.numpy() # converts the tensor to numpy array
    class_counts = np.bincount(labels) # counts number of 0s and 1s
    total_samples = len(labels)
    
    # weights for each class inverse of frequency
    weights = total_samples / (2 * class_counts) 
    return torch.FloatTensor(weights)


# where weights, epochs, and learning rates are used in training the model
def train_model(model, train_dl, dev_dl, device, n_epochs=10, lr=1e-3):
    # structured output to see performance over epochs
    print(f"\nStarting training with:")
    print(f"Number of epochs: {n_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print(f"Training batches: {len(train_dl)}")
    print(f"Validation batches: {len(dev_dl)}")
    
    
    all_labels = []
    for _, y in train_dl:
        all_labels.append(y)
    all_labels = torch.cat(all_labels, dim=0)
    
    # the two models have separate weight calculations
    severity_weights = calculate_class_weights(all_labels[:, 0])
    solved_weights = calculate_class_weights(all_labels[:, 1])
    
    print("\nWeights:")
    print(f"Severity - Not severe: {severity_weights[0]:.2f}, Severe: {severity_weights[1]:.2f}")
    print(f"Solved - Not solved: {solved_weights[0]:.2f}, Solved: {solved_weights[1]:.2f}")
    
    # uses Adam optimizer 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # binary cross entropy loss functions
    # pytorch tool we havent used in class but suggested to help make more stable numerical output
    # by 
    severity_loss = nn.BCEWithLogitsLoss(pos_weight=severity_weights[1].to(device))
    solved_loss = nn.BCEWithLogitsLoss(pos_weight=solved_weights[1].to(device))
    
    train_losses = []
    val_losses = []
    
    # training happens here !
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []
        
        # information on which epoch this iteration is on
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        for X, y_true in tqdm(train_dl, desc='Training'):
            X = X.to(device)
            y_true = y_true.to(device)
            
            # forward pass
            y_pred = model(X)
            
            # calculate weighted loss for each task
            loss_severe = severity_loss(y_pred[:, 0], y_true[:, 0].float())
            loss_solved = solved_loss(y_pred[:, 1], y_true[:, 1].float())
            total_loss = loss_severe + loss_solved
            
            # backpropogation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss.append(total_loss.item())
        
        avg_loss = np.mean(epoch_loss)
        train_losses.append(avg_loss)
        print(f"Training Loss: {avg_loss:.4f}")
        
        # evaluation happens here !!
        model.eval()
        with torch.no_grad():
            val_loss = []
            predictions = []
            actuals = []
            
            for X, y_true in tqdm(dev_dl, desc='Validation'):
                X = X.to(device)
                y_true = y_true.to(device)
                
                y_pred = model(X)
                
                loss_severe = severity_loss(y_pred[:, 0], y_true[:, 0].float())
                loss_solved = solved_loss(y_pred[:, 1], y_true[:, 1].float())
                total_loss = loss_severe + loss_solved
                
                val_loss.append(total_loss.item())
                predictions.extend(torch.sigmoid(y_pred).cpu().numpy() > 0.5)
                actuals.extend(y_true.cpu().numpy())
            
            avg_val_loss = np.mean(val_loss)
            val_losses.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            print("\nSeverity :")
            print(classification_report(actuals[:, 0], predictions[:, 0]))
            print("\nSolved :")
            print(classification_report(actuals[:, 1], predictions[:, 1]))
            
    return train_losses, val_losses