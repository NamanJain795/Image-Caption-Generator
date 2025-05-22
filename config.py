import pickle

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max_length (you can manually set it if known)
max_length = 49  # or whatever value was computed earlier

# Vocabulary size (add +1 for padding token)
vocab_size = len(tokenizer.word_index) + 1

# Load image features extracted from InceptionV3
with open("image_features.pkl", "rb") as f:
    photo_features = pickle.load(f)

# Load training captions
with open("captions.pkl", "rb") as f:
    train_descriptions = pickle.load(f)
