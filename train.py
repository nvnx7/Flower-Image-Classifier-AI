import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms, datasets, models
from collections import OrderedDict
import trainhelpers
import sys

# Get the command line arguments
args = trainhelpers.get_train_input_args()

data_directory = args.data_dir
train_dir = data_directory + '/train'
valid_dir = data_directory + '/valid'
test_dir = data_directory + '/test'

# Transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])


# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform = test_transforms)

# Using the image datasets and the trainforms, defining the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = True)
validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 32, shuffle = True)

# Get the pre-trained model & freeze the pre-trained prameters
model_str = args.arch
model = getattr(models, model_str)(pretrained = True)
for param in model.parameters():
    param.requires_grad = False

# Get the input size of the model
input_size = 0
if model_str.startswith("densenet"):
    input_size = model.classifier.in_features
elif model_str.startswith("vgg"):
        input_size = model.classifier[0].in_features
elif model_str.startswith("alexnet"):
    input_size = model.classifier[1].in_features
elif model_str.startswith("resnet"):
    input_size = model.fc.in_features
else:
    sys.exit("\n'arch' parameter got unexpected value.\
             \nOnly use variants of any of the following models:\n['densenet', 'vgg', 'alexnet', 'resnet']")    

n_hidden_units = args.hidden_units
    
classifier = nn.Sequential(OrderedDict([("fc1", nn.Linear(input_size, n_hidden_units)),
                                         ("relu", nn.ReLU()),
                                         ("drop", nn.Dropout(0.3)),
                                         ("fc2", nn.Linear(n_hidden_units, 102)),
                                         ("output", nn.LogSoftmax(dim=1))]))

if model_str.startswith("resnet"):
    model.fc = classifier
else:
    model.classifier = classifier

# Initializations
use_gpu =  args.gpu
device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")
criterion = nn.NLLLoss()
learning_rate = args.learning_rate
if model_str.startswith("resnet"):
    optimizer = optim.Adam(model.fc.parameters(), lr = learning_rate)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

# Let user decide what to choose to do if GPU not available, in case user chose GPU to be used.
if (use_gpu and not torch.cuda.is_available()):
    use_cpu = input("GPU not found, would you like to use CPU for training? (y/n)")
    if (use_cpu.lower() == "n" or use_cpu.lower() == "no"):
        sys.exit("Program terminated.")
    else:
        print("Utilising CPU for training.")
  
# Training classifier
model.to(device)
print_every = 20
steps = 0
epochs = args.epochs
for e in range(epochs):
    model.train()
    
    running_loss = 0
    for inputs, labels in trainloader:
        steps += 1
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        #Training step
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            
            # Putting model in evaluation mode
            model.eval()
        
            # Turning gradients off since no need to backpropagate for validation purpose
            with torch.no_grad():
                valid_loss, valid_accuracy = trainhelpers.validation(model, validationloader, criterion, device)
                
            print("Epoch: {} of {}    Training Loss: {:.3f}    Validation Loss: {:.3f}    Validation Accuracy: {:.2f} %"
                  .format(e+1, epochs, running_loss/print_every, valid_loss, valid_accuracy))
            
            running_loss = 0
            
            # Putting model back in training mode
            model.train()

print("\nTraining Done!")

# Saves the checkpoint
save_dir = args.save_dir
trainhelpers.save_checkpoint(model, optimizer, epochs, save_dir, train_dataset, model_str)

print("\nCheckpoint saved in: " + save_dir)