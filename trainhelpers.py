# Helping functions for training go here
import argparse
import os
import torch

# Parses and return command line arguments for train.py
def get_train_input_args():
    parser = argparse.ArgumentParser()
    cwd = os.getcwd()
    
    parser.add_argument("data_dir", type = str, help = "path to the data directory")
    parser.add_argument("--save_dir", type = str, help = "save directory for the checkpoint",
                   default = cwd)
    parser.add_argument("--arch", type = str, help = "model to use for image recognition",
                   default = "densenet121")
    parser.add_argument("--learning_rate", type = float, help = "learning rate of the model",
                    default = 0.001)
    parser.add_argument("--hidden_units", type = int, help = "number of hidden layer units",
                    default = 500)
    parser.add_argument("--epochs", type = int, help = "number for epochs",
                    default = 3)
    parser.add_argument("--gpu", action = "store_true", help = "specify wheather to use gpu to train")
    
    args = parser.parse_args()
    
    return args

# Validation pass function
def validation(model, validationloader, criterion, device):
    loss = 0
    accuracy = 0
    for inputs, labels in validationloader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model.forward(inputs)
        loss += criterion(outputs, labels)
        
        prob = torch.exp(outputs)
        equality = (prob.max(dim=1)[1] == labels)
        accuracy += equality.type(torch.FloatTensor).mean() # Mean of no. of correct predictions
        
    return loss/len(validationloader), accuracy*100/len(validationloader) # Avg. of sum of losses & % accuracy

# Saves the checkpoint i.e. trained model parameters and other needed stuff
def save_checkpoint(model, optimizer, epochs, save_dir, train_dataset, model_str):
    
    if model_str.startswith("resnet"):
        checkpoint = {'input_size': model.fc[0].in_features,
                 'output_size': model.fc[3].out_features,
                 'hidden_layers': [model.fc[i].out_features for i in (0,3)],
                 'drop_probability': model.fc[2].p,
                 'classifier_state_dict': model.fc.state_dict(),
                 'optim_state_dict': optimizer.state_dict(),
                 'epochs': epochs,
                 'map_class_to_idx': train_dataset.class_to_idx,
                 'arch': model_str}
    else:
        checkpoint = {'input_size': model.classifier[0].in_features,
                 'output_size': model.classifier[3].out_features,
                 'hidden_layers': [model.classifier[i].out_features for i in (0,3)],
                 'drop_probability': model.classifier[2].p,
                 'classifier_state_dict': model.classifier.state_dict(),
                 'optim_state_dict': optimizer.state_dict(),
                 'epochs': epochs,
                 'map_class_to_idx': train_dataset.class_to_idx,
                 'arch': model_str}

    torch.save(checkpoint, save_dir + '/' + 'checkpoint.pth')