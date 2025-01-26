from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image
image_path = "D:/animegirlpics/107925686_p0.png"
image = Image.open(image_path)

# Prepare inputs
danbooru_class_names = ["1girl",
    "solo",
    "long_hair",
    "highres",
    "breasts"]  # Replace with your tag list
inputs = clip_processor(text=danbooru_class_names, images=image, return_tensors="pt", padding=True)

# Run inference
outputs = clip_model(**inputs)
clip_probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()
print("CLIP probabilities:", clip_probs)
