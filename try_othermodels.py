import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from collections import defaultdict

# Assuming Listoftags contains the tag names for your Danbooru ResNet50 model
from classnames import Listoftags

# Load the Danbooru ResNet50 model
def load_danbooru_resnet50():
    print("Loading Danbooru ResNet50 model...")
    state_dict = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet50-13306192.pth",
        map_location=torch.device('cpu')
    )
    model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50', pretrained=False)
    model.load_state_dict(state_dict)
    print("Danbooru ResNet50 model loaded successfully.")
    return model

# Load CLIP model
def load_clip_model():
    print("Loading CLIP model...")
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model loaded successfully.")
    return model, processor

# Load Vision Transformer (ViT)
def load_vit_model():
    print("Loading Vision Transformer (ViT) model...")
    from transformers import ViTForImageClassification, ViTImageProcessor
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    print("Vision Transformer (ViT) model loaded successfully.")
    return model, processor

# Load all models
danbooru_model = load_danbooru_resnet50()
danbooru_model.eval()

clip_model, clip_processor = load_clip_model()
vit_model, vit_processor = load_vit_model()

# Preprocessing pipeline for Danbooru model
danbooru_preprocess = transforms.Compose([
    transforms.Resize(360),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
])

danbooru_class_names = Listoftags

# Analyze the predictions
def analyze_predictions(image_path, threshold=0.15):
    print(f"\nStarting analysis for: {os.path.basename(image_path)}")
    image = Image.open(image_path)

    print("Step 1: Running Danbooru ResNet50 model...")
    input_tensor = danbooru_preprocess(image).unsqueeze(0)
    with torch.no_grad():
        danbooru_probs = torch.sigmoid(danbooru_model(input_tensor)[0])
    danbooru_predictions = [
        (danbooru_class_names[i], danbooru_probs[i].item())
        for i in range(len(danbooru_probs))
        if danbooru_probs[i] > threshold
    ]
    print(f"Danbooru predictions complete. Found {len(danbooru_predictions)} tags.")

    # print("Step 2: Running CLIP model...")
    # inputs = clip_processor(text=danbooru_class_names, images=image, return_tensors="pt", padding=True)
    # outputs = clip_model(**inputs)
    # clip_probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()
    # clip_predictions = [
    #     (danbooru_class_names[i], clip_probs[i]) for i in range(len(clip_probs)) if clip_probs[i] > threshold
    # ]
    
    # Step 2: CLIP Model
    print("Step 2: Running CLIP model...")
    try:
        print("Preparing inputs for CLIP...")
        # Use only the first 1300 tags
        limited_class_names = danbooru_class_names[:1300]
        inputs = clip_processor(text=limited_class_names, images=image, return_tensors="pt", padding=True)
        print("Inputs prepared successfully.")

        print("Running inference with CLIP model...")
        outputs = clip_model(**inputs)
        print("CLIP inference complete.")

        clip_probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()
        clip_predictions = [
            (limited_class_names[i], clip_probs[i]) for i in range(len(clip_probs)) if clip_probs[i] > threshold
        ]
        print(f"CLIP predictions complete. Found {len(clip_predictions)} tags.")
    except Exception as e:
        print(f"Error during CLIP processing: {e}")
        return

    

    print("Step 3: Running Vision Transformer (ViT) model...")
    vit_inputs = vit_processor(images=image, return_tensors="pt")
    vit_outputs = vit_model(**vit_inputs)
    vit_probs = vit_outputs.logits.softmax(dim=1)[0].tolist()
    vit_predictions = [
        (danbooru_class_names[i], vit_probs[i]) for i in range(len(vit_probs)) if vit_probs[i] > threshold
    ]
    print(f"ViT predictions complete. Found {len(vit_predictions)} tags.")

    # Statistics
    danbooru_tags = {tag for tag, _ in danbooru_predictions}
    clip_tags = {tag for tag, _ in clip_predictions}
    vit_tags = {tag for tag, _ in vit_predictions}

    overlapping_tags = danbooru_tags & clip_tags & vit_tags
    unique_to_danbooru = danbooru_tags - (clip_tags | vit_tags)
    unique_to_clip = clip_tags - (danbooru_tags | vit_tags)
    unique_to_vit = vit_tags - (danbooru_tags | clip_tags)

    print("\nAnalysis Results:")
    print(f"  Overlapping Tags: {overlapping_tags}")
    print(f"  Unique to Danbooru: {unique_to_danbooru}")
    print(f"  Unique to CLIP: {unique_to_clip}")
    print(f"  Unique to ViT: {unique_to_vit}")

    # Confidence stats
    def compute_stats(predictions):
        probs = [prob for _, prob in predictions]
        return np.mean(probs), np.std(probs), np.max(probs), np.min(probs)

    danbooru_stats = compute_stats(danbooru_predictions)
    clip_stats = compute_stats(clip_predictions)
    vit_stats = compute_stats(vit_predictions)

    print(f"\nConfidence Statistics (Mean, Std, Max, Min):")
    print(f"  Danbooru: {danbooru_stats}")
    print(f"  CLIP: {clip_stats}")
    print(f"  ViT: {vit_stats}")
    print("Analysis complete.")

# Directory containing images
image_folder = "D:/animegirlpics"

# Check if directory exists
if not os.path.exists(image_folder):
    print(f"Directory {image_folder} not found. Please check the path.")
else:
    # Iterate over all image files in the folder
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(image_folder, image_file)
            print(f"Processing image: {image_file}...")
            analyze_predictions(image_path)
