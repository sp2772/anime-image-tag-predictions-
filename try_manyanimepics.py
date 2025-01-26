import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from classnames import Listoftags  # Assuming you have this module with the list of tags.

# Function to load model weights and map them to the CPU
def load_danbooru_resnet50():
    # Download the model weights file locally
    state_dict = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet50-13306192.pth",
        map_location=torch.device('cpu')  # Map to CPU
    )
    # Load the model architecture
    model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50', pretrained=False)
    model.load_state_dict(state_dict)
    return model

# Load the model
model = load_danbooru_resnet50()
model.eval()

# Preprocessing pipeline for images
preprocess = transforms.Compose([
    transforms.Resize(360),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
])

class_names = Listoftags

def process_and_display(image_path, threshold=0.15):
    # Open the image
    input_image = Image.open(image_path)
    
    # Preprocess the image
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_batch)
    
    # Get probabilities
    probs = torch.sigmoid(output[0])
    
    # Get predictions above the threshold
    tmp = probs[probs > threshold]
    inds = probs.argsort(descending=True)
    
    # Display image
    plt.imshow(input_image)
    plt.grid(False)
    plt.axis('off')
     
    
    # Display predictions as text
    txt = f"Predictions for {os.path.basename(image_path)} (threshold > {threshold}):\n"
    for i in inds[:len(tmp)]:
        txt += f"{class_names[i]}: {probs[i].item():.4f}\n"
    print(txt,'\n')
    tagforimg=(txt.split('\n'))[1:]      #list of values like "red_eyes: 0.5675"
    plt.text(input_image.size[0]*1.05, input_image.size[1]*0.85, txt)
    
    # Show the results
    plt.tight_layout()
    plt.show()

# Directory containing images
image_folder = "D:/animegirlpics"

# Iterate over all image files in the folder
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
        image_path = os.path.join(image_folder, image_file)
        process_and_display(image_path)
