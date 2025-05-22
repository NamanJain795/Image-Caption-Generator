import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import gc

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image

# Limit GPU memory growth if using GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU setup error:", e)

# Load pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')
model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

# Image directory
image_dir = r'C:\Users\naman\OneDrive\Desktop\temp image captioning\train2014'  # adjust if needed: 'images/train2014'
output_file = 'image_features.pkl'

# Output dictionary
features = {}

# Process each image
image_list = os.listdir(image_dir)

for img_name in tqdm(image_list):
    img_path = os.path.join(image_dir, img_name)

    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features
        feature = model.predict(x)
        features[img_name] = feature

        # Clear memory
        del img, x, feature
        gc.collect()

    except UnidentifiedImageError:
        print(f"Skipping corrupt image: {img_name}")
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        continue

# Save features
with open(output_file, 'wb') as f:
    pickle.dump(features, f)

print(f"\n✅ Features extracted and saved for {len(features)} images.")
