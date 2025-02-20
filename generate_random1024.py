import numpy as np

# Generate random token sequences
num_sequences = 1000
sequence_length = 1024
vocab_size = 10000

# Create random token IDs within the vocabulary range
token_ids = np.random.randint(
    low=0,
    high=vocab_size,
    size=(num_sequences * sequence_length),
    dtype=np.uint32
)

# Write token IDs to disk
data_mmap = np.memmap(
    "random1024.npy",
    mode="w+",
    dtype=np.uint32,
    shape=(num_sequences * sequence_length,)
)
data_mmap[:] = token_ids
data_mmap.flush()