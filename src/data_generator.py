import numpy as np
from keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import tensorflow as tf
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, descriptions, features, vocab, max_length, vocab_size, batch_size=64):
        self.descriptions = descriptions
        self.features = features
        self.vocab = vocab
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.image_ids = list(descriptions.keys())
        self.indices = np.arange(len(self.image_ids))

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]
        X1, X2, y = list(), list(), list()

        for img_id in batch_ids:
            captions = self.descriptions[img_id]
            image_feature = self.features[img_id].squeeze()
            for caption in captions:
                seq = self.vocab.caption_to_seq(caption)
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(image_feature)
                    X2.append(in_seq)
                    y.append(out_seq)

        # Convert lists to numpy arrays
        X1 = np.array(X1, dtype=np.float32)
        X2 = np.array(X2, dtype=np.int32)
        y = np.array(y, dtype=np.float32)

        # ✅ Return a tuple of tuples, not a list
        return (X1, X2), y

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        self.indices = np.random.permutation(len(self.image_ids))

