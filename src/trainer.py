import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os


# training script + quick evaluation
# handles weights and imbalances - sets up the optimizer

# this is where parameter changes can occur !!

def calculate_class_weights(labels):
    labels = labels.numpy()  
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    weights = total_samples / (2 * class_counts)
    scaled_weights = np.sqrt(weights)
    return torch.FloatTensor(scaled_weights)


# where weights, epochs, and learning rates are used in training the model
def train_model(model, train_dl, dev_dl, device, n_epochs=7, lr=1e-3):
    output_dir = 'confusion_matrices'
    os.makedirs(output_dir, exist_ok=True)
    # structured output to see performance over epochs
    with open('output.txt', 'w') as file:
        output = f"\nStarting training with:\n" + \
                 f"Number of epochs: {n_epochs}\n" + \
                 f"Learning rate: {lr}\n" + \
                 f"Device: {device}\n" + \
                 f"Training batches: {len(train_dl)}\n" + \
                 f"Validation batches: {len(dev_dl)}\n"
        print(output)
        file.write(output)
    
    
        all_labels = []
        for _, y in train_dl:
            all_labels.append(y)
        all_labels = torch.cat(all_labels, dim=0)
        
        # the two models have separate weight calculations
        severity_weights = calculate_class_weights(all_labels[:, 0])
        solved_weights = calculate_class_weights(all_labels[:, 1])
        
        output = "\nWeights:\n" + \
                 f"Severity - Not severe: {severity_weights[0]:.2f}, Severe: {severity_weights[1]:.2f}\n" + \
                 f"Solved - Not solved: {solved_weights[0]:.2f}, Solved: {solved_weights[1]:.2f}\n"
        print(output)
        file.write(output)
        
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
            file.write(f"\nEpoch {epoch+1}/{n_epochs}\n")
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
            print(f"Training Loss: {avg_loss:.4f}")
            file.write(f"Training Loss: {avg_loss:.4f}\n")
        
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
                output = f"Validation Loss: {avg_val_loss:.4f}\n"
                print(output)
                file.write(output)
                
                predictions = np.array(predictions)
                actuals = np.array(actuals)

                cm_severity = confusion_matrix(actuals[:, 0], predictions[:, 0])
                cm_solved = confusion_matrix(actuals[:, 1], predictions[:, 1])
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_severity, display_labels=['Not Severe', 'Severe'])
                disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_solved, display_labels=['Not Solved', 'Solved'])
                
                disp1.plot(ax=ax[0], cmap='Blues')
                ax[0].set_title('Confusion Matrix: Severity')
                
                disp2.plot(ax=ax[1], cmap='Blues')
                ax[1].set_title('Confusion Matrix: Solved')
                
                plt.savefig(os.path.join(output_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
                plt.close(fig)
                
                severity_report = classification_report(actuals[:, 0], predictions[:, 0])
                solved_report = classification_report(actuals[:, 1], predictions[:, 1])
                print("\nSeverity :")
                print(severity_report)
                print("\nSolved :")
                print(solved_report)

                file.write("\nSeverity :\n")
                file.write(severity_report)
                file.write("\nSolved :\n")
                file.write(solved_report)
            
    return train_losses, val_losses