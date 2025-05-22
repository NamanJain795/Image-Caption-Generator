import pickle
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, add
from data_generator import DataGenerator
from vocabulary import Vocabulary

# Config
BATCH_SIZE = 64
EPOCHS = 10
MAX_LENGTH = 34  # Adjust based on your dataset
EMBEDDING_DIM = 256
UNITS = 256

# Load data
with open(r'C:\Users\naman\OneDrive\Desktop\temp image captioning\src\captions.pkl', 'rb') as f:
    descriptions = pickle.load(f)

with open(r'C:\Users\naman\OneDrive\Desktop\temp image captioning\src\image_features.pkl', 'rb') as f:
    features = pickle.load(f)

with open(r'C:\Users\naman\OneDrive\Desktop\temp image captioning\src\vocab.pkl', 'rb') as f:
    vocab: Vocabulary = pickle.load(f)

vocab_size = len(vocab)

# Data generator
train_generator = DataGenerator(descriptions, features, vocab, MAX_LENGTH, vocab_size, batch_size=BATCH_SIZE)

# Define the model
def define_model(vocab_size, max_length):
    # Image feature extractor model input
    inputs1 = tf.keras.Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence model input
    inputs2 = tf.keras.Input(shape=(max_length,))
    se1 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder (combine)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Initialize and train the model
model = define_model(vocab_size, MAX_LENGTH)
steps = len(train_generator)

model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=steps, verbose=1)

# Save model
model.save('../caption_model.keras')
print("✅ Model saved as caption_model.h5")
