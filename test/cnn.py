import torch.nn as nn
import torch.nn.functional as F
import torch


## Test interpolate and argmax and embedding.
vocab_size = 10
embedding_size = 3
embedding = nn.Embedding(vocab_size, embedding_size)

print(f"Embedding weights: {embedding.weight}")
print(f"Embedding weights data: {embedding.weight.data}")
print(f"Embedding weights dtype: {embedding.weight.data.dtype}")

def interpolate(x, size: tuple, mode='nearest'):
    return F.interpolate(x, size=size, mode=mode)

input_sample = torch.rand((embedding_size, 64, 64))  # Random input tensor of shape (embedding_size, 64, 64)
H = W = len(input_sample[1])
interpolated_sample = interpolate(input_sample.unsqueeze(0), size=(2, 2))  # Interpolate to (32, 32)
print(f"Input sample: {input_sample}")
print(f"Input sample dtype: {input_sample.dtype}")
print(f"Size (2,2) interpolation size: {interpolate(input_sample.unsqueeze(0), size=(2, 2), mode='area').shape}")
print("Interpolated sample (2,2):", interpolated_sample)


input_sample = input_sample.permute(1, 2, 0)  # Reshape to (2, 3, 1)
input_sample_features = input_sample.reshape(-1, embedding_size)  # Reshape to (H x W, embedding_size)
index_N = torch.argmax(input_sample_features @ embedding.weight.data.T, dim=1)  # Get the index of the max value along the embedding dimension for each feature vector.


print(f"Index N shape: {index_N.shape}")
print(f"Index N: {index_N}")


embedded_feature = embedding(index_N.view(1, H, W)) # Goes to each value of index_N and look up the embedding of that index in the embedding matrix.
print(f"Embedded size: {embedded_feature.shape}")
print(f"Embedded feature: {embedded_feature}")