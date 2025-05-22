# src/vocabulary.py

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.word_freq = {}
        self.idx = 4

    def build_vocabulary(self, sentence_list):
        self.word2idx["<pad>"] = 0
        self.word2idx["<start>"] = 1
        self.word2idx["<end>"] = 2
        self.word2idx["<unk>"] = 3

        self.idx2word[0] = "<pad>"
        self.idx2word[1] = "<start>"
        self.idx2word[2] = "<end>"
        self.idx2word[3] = "<unk>"

        idx = 4

        for sentence in sentence_list:
            for word in sentence.split():
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def numericalize(self, sentence):
        return [
            self.word2idx.get(word, self.word2idx["<UNK>"])
            for word in sentence.split()
        ]

    def __len__(self):
        return len(self.word2idx)

    def caption_to_seq(self, caption):
        unk_idx = self.word2idx.get("<unk>", 0)  # Default to 0 if <unk> not in vocab
        return [self.word2idx.get(word, unk_idx) for word in caption.split()]

