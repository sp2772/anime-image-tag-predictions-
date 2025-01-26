import json
import urllib.request

# Load the existing JSON file
with urllib.request.urlopen("https://github.com/RF5/danbooru-pretrained/raw/master/config/class_names_6000.json") as url:
    class_names = json.loads(url.read().decode())

# Define the desired tags
desired_tags = {
    "hair_colors": ["red hair", "blue hair", "black hair", "green hair", "blonde hair", "pink hair", "purple hair", "brown hair", "white hair", "grey hair", "silver hair"],
    "eye_colors": ["red eyes", "blue eyes", "black eyes", "green eyes", "hazel eyes", "amber eyes", "brown eyes", "yellow eyes", "violet eyes"],
    "skin_colors": ["fair skin", "tan skin", "dark skin"],
    "body_features": ["tall", "short", "smiling", "angry", "happy", "sad", "pretty", "ugly", "face", "blush", "freckles"],
    "dress_types": ["kimono", "school uniform", "swimsuit", "casual wear", "maid outfit", "armor", "nurse outfit", "cheerleader outfit"],
}

# Combine all desired tags into a single list
desired_tags_flat = [tag.replace(" ","_") for tags in desired_tags.values() for tag in tags]

# Find missing tags
missing_tags = [tag.replace(" ","_") for tag in desired_tags_flat if tag.replace(" ","_") not in class_names]

# Add missing tags to the existing class names
updated_class_names = class_names + missing_tags

# Save the updated class names to a new JSON file
with open("updated_class_names.json", "w") as outfile:
    json.dump(updated_class_names, outfile, indent=4)

print(f"Updated JSON file saved with {len(missing_tags)} new tags.")
