import os
import tiktoken
import numpy as np
import pandas as pd


# Read plain text file directly
with open('data/text/Module_9.txt', 'r', encoding='utf-8') as f:
    data = f.read()

# Split the text 90/10 for train and validation
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Save the split files
with open('data/text/train.txt', 'w', encoding='utf-8') as f:
    f.write(train_data)

with open('data/text/val.txt', 'w', encoding='utf-8') as f:
    f.write(val_data)

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
