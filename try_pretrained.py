
import torch
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
import json
import urllib, urllib.request


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

# Preprocess the image
input_image = Image.open("D:/PythonProjects/generated_image.png")
preprocess = transforms.Compose([
    transforms.Resize(360),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(input_batch)

# Get probabilities
probs = torch.sigmoid(output[0])

from classnames import Listoftags
class_names=Listoftags



# Get class names
# with urllib.request.urlopen("https://github.com/RF5/danbooru-pretrained/raw/master/config/class_names_6000.json") as url:
#     class_names = json.loads(url.read().decode())


# Plot image
plt.imshow(input_image)
plt.grid(False)
plt.axis('off')

def plot_text(thresh=0.15):
    tmp = probs[probs > thresh]
    inds = probs.argsort(descending=True)
    txt = 'Predictions with probabilities above ' + str(thresh) + ':\n'
    for i in inds[0:len(tmp)]:
        txt += class_names[i] + ': {:.4f} \n'.format(probs[i].cpu().numpy())
    plt.text(input_image.size[0]*1.05, input_image.size[1]*0.85, txt)

plot_text()
plt.tight_layout()
plt.show()
