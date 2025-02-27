import torch

def hellinger(p, q):
    """
    Compute the Hellinger distance between two probability distributions.
    """
    return torch.sqrt(0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2))