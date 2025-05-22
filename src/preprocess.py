# src/preprocess.py

import json
import pickle
from vocabulary import Vocabulary  # Make sure this import matches your folder

ANNOTATION_FILE = "../annotations/captions_train2014.json"
CAPTION_OUTPUT = "captions.pkl"
VOCAB_OUTPUT = "vocab.pkl"

def load_captions(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    captions_dict = {}
    for annot in annotations['annotations']:
        img_id = annot['image_id']
        img_filename = f"COCO_train2014_{img_id:012}.jpg"
        caption = annot['caption'].lower().strip()
        captions_dict.setdefault(img_filename, []).append(caption)

    return captions_dict

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def main():
    print("Loading captions...")
    captions_dict = load_captions(ANNOTATION_FILE)
    save_pickle(captions_dict, CAPTION_OUTPUT)
    print(f"Saved captions to {CAPTION_OUTPUT}")

    print("Building vocabulary...")
    all_captions = []
    for caption_list in captions_dict.values():
        all_captions.extend([f"startseq {cap} endseq" for cap in caption_list])

    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_captions)
    save_pickle(vocab, VOCAB_OUTPUT)
    print(f"Saved vocabulary to {VOCAB_OUTPUT}")
    print("Vocabulary size:", len(vocab))

if __name__ == "__main__":
    main()
