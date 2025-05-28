import torch

def dictionary_constructor(input_dim, reduced_dim):
    probabilities = torch.empty(reduced_dim).uniform_(0, 1)
    probabilities = probabilities.reshape((reduced_dim, 1))
    probabilities = probabilities.expand((reduced_dim, input_dim))
    dictionary = torch.bernoulli(probabilities).float()
    return dictionary