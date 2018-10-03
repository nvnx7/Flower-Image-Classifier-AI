# Helping functions for prediction go here
import argparse
import numpy as np
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
from PIL import Image

# Parses and return command line arguments for predict.py
def get_predict_input_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument("img_path", type = str, help = "path of the image to predict")
    parser.add_argument("checkpoint", type = str, help = "path to the checkpoint of trained model")
    parser.add_argument("category_names", type = str, help = "path to the mapping file that maps category to names")
    parser.add_argument("--top_k", type = int, default = 5, help = "return the top k results")
    parser.add_argument("--gpu", action = "store_true", help = "wheather to use gpu for inference or not")
    
    args = parser.parse_args()
    return args

# Loads a checkpoint and rebuilds the model
def load_model(path, cpu_is_used):
    if (cpu_is_used):
        checkpoint = torch.load(path, map_location="cpu")
    else:
        checkpoint = torch.load(path, map_location="cuda:0")
    classifier = nn.Sequential(OrderedDict([("fc1", nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0])),
                                         ("relu", nn.ReLU()),
                                         ("drop", nn.Dropout(checkpoint['drop_probability'])),
                                         ("fc2", nn.Linear(checkpoint['hidden_layers'][0], checkpoint['hidden_layers'][1])),
                                         ("output", nn.LogSoftmax(dim=1))]))
    
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    model = getattr(models, checkpoint['arch'])(pretrained = True)
    if checkpoint['arch'].startswith("resnet"):
        model.fc = classifier
    else:
        model.classifier = classifier
        
    model.class_to_idx = checkpoint['map_class_to_idx']
    
    return model

# Normalizes the image array expecting color channel to be first dimension
def normalize_image_array(img_array, means, std):
    ''' Encodes the color channels between 0 and 1 & normalizes the 
        img_array (expecting color channel to be first dimension)
        given the means - 'means' and standard deviations - 'std'
    '''
    img_array = img_array/255
    # For each color channel
    for i in range(img_array.shape[0]):
        img_array[i] = (img_array[i] - means[i])/std[i]
        
    return img_array

# Preprocessing the PIL image to make it fit for input to the model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_img = Image.open(image)
    
    # Resizing while preserving aspect ratio
    width, height = pil_img.size
    if width <= height:
        new_width = 256
        new_height = int((new_width*height)/width)
    else:
        new_height = 256
        new_width = int((new_height*width)/height)
    pil_img = pil_img.resize(size = (new_width, new_height))
    
    # Center cropping 224x224
    size = pil_img.size
    pil_img = pil_img.crop(( size[0]//2 - 112,
                             size[1]//2 - 112,
                             size[0]//2 + 112,
                             size[1]//2 + 112))
    
    # Converting to numpy array and noramlizing
    np_img = np.array(pil_img, dtype = 'float')
    np_img = np_img.T
    np_img = normalize_image_array(np_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    return np_img

# Predicts the image using given model
def predict(image_path, model, topk, device):
    ''' Predict the top topk class (or classes) of an image using a trained deep learning model.
        Returns the top topk probabilities and corresponding classes.
    '''
    model.eval()
    np_img = process_image(image_path)
    
    tens_img = torch.from_numpy(np_img)
    tens_img = tens_img.to(device)
    tens_img.unsqueeze_(0)
    tens_img = tens_img.type(torch.FloatTensor)
    
    with torch.no_grad():
        output = model.forward(tens_img)
        
    prob = torch.exp(output)
    topk_values, indices = prob.topk(topk)
    
    # Get mapping of class to index and reverse the mapping
    indices = indices.numpy().squeeze()
    class_to_index = model.class_to_idx
    index_to_class = {v:k for k,v in class_to_index.items()}
    classes = [int(index_to_class[index]) for index in indices]    # Classes of top k flowers predicted
    
    topk_values = topk_values.numpy().squeeze()
    
    return topk_values, classes
