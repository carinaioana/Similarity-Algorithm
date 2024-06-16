import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle
from tqdm import tqdm
import boto3
from io import BytesIO

# Step 1: Load pre-trained CNN model and set it to evaluation mode
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
model = model.to(device)
model.eval()

# Step 2: Define a function to extract features from an image
def extract_features(image):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        features = model(image).squeeze().cpu().numpy()

    return features

# AWS S3 bucket details
s3_bucket_name = 'my-images-bucket38'
s3 = boto3.client('s3',
                  aws_access_key_id='AKIAXPDNKTDCE7UJT3VJ',
                  aws_secret_access_key='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',
                  region_name='eu-north-1')

# List to store image file paths
image_paths = []

# Fetch the list of images from the S3 bucket
for main_class in ["cats", "dogs"]:
    prefix = f"{main_class}/"
    response = s3.list_objects_v2(Bucket=s3_bucket_name, Prefix=prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(obj['Key'])
    # Limit to top 1000 images per class
    image_paths = image_paths[:1000]

print(f"Collected {len(image_paths)} image paths.")

# Load existing data
if os.path.exists('image.pkl'):
    with open('image.pkl', 'rb') as f:
        existing_image_paths = pickle.load(f)
else:
    existing_image_paths = []

# Identify new images
new_image_paths = [path for path in image_paths if path not in existing_image_paths]

# Load existing features and names
if os.path.exists('feat.pkl'):
    with open('feat.pkl', 'rb') as f:
        dataset_features = pickle.load(f)
else:
    dataset_features = []

if os.path.exists('names.pkl'):
    with open('names.pkl', 'rb') as f:
        dataset_names = pickle.load(f)
else:
    dataset_names = []

# Create the 'FEAT' directory if it doesn't exist
os.makedirs('FEAT', exist_ok=True)

# Extract features for new images and save them
for path in tqdm(new_image_paths):
    response = s3.get_object(Bucket=s3_bucket_name, Key=path)
    image_data = response['Body'].read()
    image = Image.open(BytesIO(image_data))
    features = extract_features(image)

    # Ensure the path is correctly formatted
    path_parts = path.split('/')
    if len(path_parts) > 1:
        class_name = path_parts[0]
    else:
        class_name = 'unknown'

    os.makedirs(os.path.join('FEAT', class_name), exist_ok=True)

    # Save features to a pickle file
    filename = os.path.join('FEAT', class_name, os.path.splitext(os.path.basename(path))[0] + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(features, f)

    # Update lists
    dataset_features.append(features)
    dataset_names.append(os.path.splitext(os.path.basename(path))[0])
    existing_image_paths.append(path)

# Save the updated image paths
with open('image.pkl', 'wb') as f:
    pickle.dump(existing_image_paths, f)

# Save the updated features
with open('feat.pkl', 'wb') as f:
    pickle.dump(dataset_features, f)

# Save the updated names
with open('names.pkl', 'wb') as f:
    pickle.dump(dataset_names, f)

print(f"Updated with {len(new_image_paths)} new images.")
