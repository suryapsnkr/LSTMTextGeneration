"""
Generative AI Engineer Task – LSTM Text Generation
------------------------------------------------
Character‑level LSTM with improved architecture and larger dataset subset.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import string
import random

# -------------------------------
# 1. Load & preprocess text (use 1M chars for good quality)
# -------------------------------
def load_text(file_path, max_chars=1_000_000):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text[:max_chars]

file_path = 'shakespeare.txt'
text = load_text(file_path, max_chars=1_000_000)   # 1M characters
print(f"Using {len(text):,} characters")

chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
print(f"Vocabulary: {vocab_size}")

# -------------------------------
# 2. Create sequences (longer context)
# -------------------------------
seq_length = 100          # better context than 50
step = 3                  # smaller step → more training samples

sequences = []
next_chars = []
for i in range(0, len(text) - seq_length, step):
    seq = text[i:i+seq_length]
    target = text[i+seq_length]
    seq_int = [char_to_idx[ch] for ch in seq]
    sequences.append(seq_int)
    next_chars.append(char_to_idx[target])

X = np.array(sequences, dtype=np.int32)
y = np.array(next_chars, dtype=np.int32)
print(f"Total sequences: {X.shape[0]:,}")

# Train/validation split
split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# -------------------------------
# 3. Build a deeper LSTM model
# -------------------------------
model = Sequential()
model.add(Embedding(vocab_size, 256, input_length=seq_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()

# -------------------------------
# 4. Train with early stopping
# -------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('lstm_shakespeare.keras', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=128,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------
# 5. Text generation
# -------------------------------
def generate_text(seed, model, char_to_idx, idx_to_char, seq_length,
                  num_chars=200, temperature=0.8):
    seed = seed[-seq_length:].rjust(seq_length, ' ')
    generated = seed
    for _ in range(num_chars):
        seed_int = [char_to_idx.get(ch, 0) for ch in seed]
        pred = model.predict(np.array([seed_int]), verbose=0)[0]
        # Temperature sampling
        pred = np.log(pred + 1e-8) / temperature
        exp_pred = np.exp(pred)
        pred = exp_pred / np.sum(exp_pred)
        next_idx = np.random.choice(range(vocab_size), p=pred)
        next_char = idx_to_char[next_idx]
        seed = seed[1:] + next_char
        generated += next_char
    return generated

# -------------------------------
# 6. Sample outputs
# -------------------------------
seeds = [
    "to be or not to be",
    "the course of true love",
    "all the world's a stage"
]

print("\n--- Generated Text ---")
for seed in seeds:
    print(f"\nSeed: '{seed}'")
    gen = generate_text(seed, model, char_to_idx, idx_to_char,
                        seq_length, num_chars=200, temperature=0.8)
    print(f"Generated:\n{gen}\n")