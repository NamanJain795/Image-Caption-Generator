import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import h5py
from tqdm import tqdm

# ✅ Update path to your MS COCO train images
IMAGE_DIR = "C:/Users/naman/OneDrive/Desktop/temp image captioning/train2014"
OUTPUT_PATH = "image_features.h5"

# Load InceptionV3 model (without final classification layer)
model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)  # output from the second last layer

def extract(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model_new.predict(x)
    feature = np.reshape(feature, feature.shape[1])
    return feature

# Save features in HDF5 file
with h5py.File(OUTPUT_PATH, 'w') as h:
    for img_name in tqdm(os.listdir(IMAGE_DIR)):
        if not img_name.endswith(".jpg"):
            continue
        image_id = os.path.splitext(img_name)[0].split('_')[-1]  # e.g., 'COCO_train2014_000000581929.jpg' ➜ '581929'
        file_path = os.path.join(IMAGE_DIR, img_name)
        try:
            feature = extract(file_path)
            h.create_dataset(image_id, data=feature)
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
