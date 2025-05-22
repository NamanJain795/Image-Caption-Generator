import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from vocabulary import Vocabulary

MAX_LENGTH = 34
MODEL_PATH = 'caption_model.keras'
VOCAB_PATH = r'C:\Users\naman\OneDrive\Desktop\temp image captioning\src\vocab.pkl'
TEST_IMAGE_PATH = r"C:\Users\naman\OneDrive\Desktop\temp image captioning\test2014\COCO_test2014_000000003257.jpg"
USE_TOP_K = False

print("Loading caption model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

print("Loading vocabulary...")
with open(VOCAB_PATH, 'rb') as f:
    vocab: Vocabulary = pickle.load(f)
index_to_word = vocab.idx2word
word_to_index = vocab.word2idx
print("Vocabulary loaded.")

def get_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

def extract_features(filename, model):
    img = image.load_img(filename, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten()

def generate_caption(model, image_features, vocab, max_length=34, top_k=5):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = [vocab.word2idx.get(word, vocab.word2idx.get('<unk>', 3)) for word in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

        preds = model.predict([np.array([image_features]), sequence], verbose=0)[0]

        if USE_TOP_K:
            top_indices = preds.argsort()[-top_k:][::-1]
            top_probs = preds[top_indices]
            top_probs /= np.sum(top_probs)
            yhat = np.random.choice(top_indices, p=top_probs)
        else:
            yhat = np.argmax(preds)

        word = vocab.idx2word.get(yhat, '<unk>')

        if word in ['<end>', '<pad>', '<unk>']:
            break

        in_text += ' ' + word

    final_caption = in_text.replace('<start>', '').strip()
    return final_caption


if __name__ == "__main__":
    print(f"Extracting features from: {TEST_IMAGE_PATH}")
    feature_model = get_feature_extractor()
    image_features = extract_features(TEST_IMAGE_PATH, feature_model)

    print("Generating caption...")
    caption = generate_caption(model, image_features, vocab, MAX_LENGTH)
    print("Caption:", caption)
