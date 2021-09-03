import torch
from torch.nn import functional as F

# Sampling from top-k follows the idea of: https://arxiv.org/abs/1904.09751
# One day I will implement sampling from top-p also: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317


# Credits to Karpathy https://github.com/karpathy/minGPT/blob/master/mingpt/utils.py
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def generate(a_i, temperature=1.0, top_k=None, sample=False):
    # pluck the logits at the final step and scale by temperature
    logits = a_i / temperature
    # optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution or take the most likely
    if sample:
        ix = torch.multinomial(probs, num_samples=1)
    else:
        _, ix = torch.topk(probs, k=1, dim=-1)
    return ix


# Credits to Farbod https://github.com/farbodtaymouri/MLMME/blob/main/network.py
def label_smooth_gumbel_sampling(t):
    t[t == 1] = 0.9
    t[t == 0] = 0.1 / (t.size(2) - 1)
    t = F.gumbel_softmax(t, dim=-1, tau=0.001)
    return t
