import numpy as np

raw = np.fromfile("../../big_jan5.bin", dtype=np.uint8)

SPI_CHUNK_BYTES = 2048

n = raw.size // SPI_CHUNK_BYTES
raw = raw[:n * SPI_CHUNK_BYTES]

# Compare first 8 bytes of each chunk
headers = raw.reshape(n, SPI_CHUNK_BYTES)[:, :8]

# Count how many unique headers exist
uniq = np.unique(headers, axis=0)
print("Unique first-8-byte patterns:", len(uniq))

# Show first few
print(uniq[:5])