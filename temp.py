import matplotlib.pyplot as plt

metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
lstm_scores = [0.47, 0.32, 0.20, 0.10, 0.19, 0.40, 0.35]
ref_scores = [0.71, 0.62, 0.53, 0.42, 0.31, 0.58, 0.94]

x = range(len(metrics))
width = 0.35

plt.bar([i - width/2 for i in x], lstm_scores, width, label='LSTM Model')
plt.bar([i + width/2 for i in x], ref_scores, width, label='Reference Model')
plt.xticks(x, metrics)
plt.ylabel('Scores')
plt.title('Quantitative Comparison of Captioning Models')
plt.legend()
plt.tight_layout()
plt.show()
