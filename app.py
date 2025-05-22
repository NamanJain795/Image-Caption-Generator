import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vocabulary import Vocabulary

# Constants
MAX_LENGTH = 34
MODEL_PATH = 'caption_model.keras'
VOCAB_PATH = r'C:\Users\naman\OneDrive\Desktop\temp image captioning\src\vocab.pkl'

# Load model and vocabulary
@st.cache_resource
def load_model_and_vocab():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    return model, vocab

model, vocab = load_model_and_vocab()

# Feature extractor
@st.cache_resource
def get_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model_feat = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model_feat

feature_extractor = get_feature_extractor()

# Functions
def extract_features(uploaded_file, feature_model):
    img = image.load_img(uploaded_file, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = feature_model.predict(x, verbose=0)
    return features.flatten()

def generate_caption(model, image_features, vocab, max_length=34):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = [vocab.word2idx.get(word, vocab.word2idx.get('<unk>', 3)) for word in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

        preds = model.predict([np.array([image_features]), sequence], verbose=0)[0]
        yhat = np.argmax(preds)
        word = vocab.idx2word.get(yhat, '<unk>')

        if word in ['<end>', '<pad>', '<unk>']:
            break

        in_text += ' ' + word

    final_caption = in_text.replace('<start>', '').strip()
    return final_caption

# Streamlit UI
st.title("🖼️ Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("Generating caption...")

    image_features = extract_features(uploaded_file, feature_extractor)
    caption = generate_caption(model, image_features, vocab, MAX_LENGTH)

    st.success("Caption Generated:")
    st.subheader(caption)
