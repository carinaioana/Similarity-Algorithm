import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import faiss
import numpy as np
import pickle
from matplotlib import pyplot as plt
import os
import boto3
from io import BytesIO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# AWS S3 bucket details
s3_bucket_name = 'my-images-bucket38'
s3 = boto3.client('s3',
                  aws_access_key_id='AKIAXPDNKTDCE7UJT3VJ',
                  aws_secret_access_key='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',
                  region_name='eu-north-1')

# Input the image and label
input_image = Image.open("test-images/6.jpg")
input_label = "bengal"  # Assuming you have the ground truth label for the input image

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


# Load dataset paths and features
with open('image.pkl', 'rb') as f:
    dataset_paths = pickle.load(f)

with open('feat.pkl', 'rb') as f:
    dataset_features = pickle.load(f)

with open('names.pkl', 'rb') as f:
    dataset_names = pickle.load(f)

name_to_index = {name: index for index, name in enumerate(dataset_names)}

def sort_key(path):
    name = os.path.basename(path).split('.')[0]
    return name_to_index[name]

# Sort dataset_paths using the custom key function
sorted_dataset_paths = sorted(dataset_paths, key=sort_key)
dataset_paths = sorted_dataset_paths

# Extract breed names (folder names) from the paths
dataset_labels = []
for path in dataset_paths:
    parts = path.split("/")
    if len(parts) >= 3:
        dataset_labels.append(parts[1])  # Breed name is the folder name, which is the third part
    else:
        print(f"Unexpected path format: {path}")
        dataset_labels.append("unknown")  # Assign a default value if the format is unexpected

dataset_features = np.array(dataset_features)
index = faiss.IndexFlatL2(dataset_features.shape[1])
index.add(dataset_features)

# Function to read image from S3
def read_image_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    image_data = response['Body'].read()
    image = Image.open(BytesIO(image_data))
    return image

def find_similar_images(input_image, input_label, k=5):
    input_features = extract_features(input_image)
    distances, indices = index.search(np.array([input_features]), k)

    similar_images = []
    relevant_count = 0
    for idx in indices[0]:
        similar_image_path = dataset_paths[idx]
        similar_label = dataset_labels[idx]
        if similar_label == input_label:
            relevant_count += 1
        similar_images.append((similar_image_path, similar_label))

    precision = relevant_count / k

    return similar_images, precision

input_label = input_label.lower()
similar_images, precision = find_similar_images(input_image, input_label, k=2)

print(f"Precision@{len(similar_images)}: {precision:.4f}")

print("\nSimilar images:")
for path, label in similar_images:
    print(f"Path: {path}, Label: {label}")

x = [input_image]
for path, _ in similar_images:
    s3_image = read_image_from_s3(s3_bucket_name, path)
    x.append(s3_image)

images = x

# Define the size of the grid based on the number of images
num_images = len(images)
rows = 2  # Number of rows in the grid
cols = (num_images + 1) // rows  # Calculate the number of columns based on the number of images

# Create a figure and axis
fig, axes = plt.subplots(rows, cols, figsize=(40, 15))

# Flatten the axes if it's a single row or column
axes = axes.flatten()

# Plot each image
for i, image in enumerate(images):
    ax = axes[i]
    ax.imshow(image)
    ax.axis('off')  # Turn off axis labels
    if i == 0:
        ax.set_title(input_label, fontsize=70)
    else:
        ax.set_title(f'{similar_images[i - 1][1]}', fontsize=50)

# Hide any unused subplots
for ax in axes[num_images:]:
    ax.axis('off')

# Add a super title to the figure to show precision
if precision != 0:
    fig.suptitle(f'Precision: {precision:.4f}', fontsize=50)

plt.tight_layout()
plt.show()
