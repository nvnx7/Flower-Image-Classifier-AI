import predicthelpers
import json
import torch
import sys

# Get the command line arguments
args = predicthelpers.get_predict_input_args()

# Load the class to name mapping file
category_names = args.category_names
with open(category_names, 'r') as file:
    cat_to_name = json.load(file)



# Set the device to use
use_gpu = args.gpu
device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")

use_cpu = not use_gpu

# Let user decide what to choose to do if GPU not available, in case user chose GPU to be used.
if (use_gpu and not torch.cuda.is_available()):
    response = input("GPU not found, would you like to use CPU for prediction? (y/n)")
    if (response.lower() == "n" or response.lower() == "no"):
        sys.exit("Program terminated.")
    else:
        use_cpu = True
        print("Utilising CPU for prediction.")
        
# Load the model with checkpoint
checkpoint_path = args.checkpoint

model = predicthelpers.load_model(checkpoint_path, use_cpu)

# Predict the image & give top top_k results with their indices
top_k = args.top_k
image_path = args.img_path
topk_v, classes = predicthelpers.predict(image_path, model, top_k, device)    

# Actual names of those flowers
labels = [cat_to_name[str(index)] for index in classes]

# Printing the results
print ("\nPredicted flower: {} with probability {:.3f}.".format(labels[0], topk_v[0]))

print("\nTop {} predictions:".format(top_k))
print("Flower             Probability")
print("-"*32)

for i in range(top_k):
    print("{}      -      {:.3f}".format(labels[i], topk_v[i]))