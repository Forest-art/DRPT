from sklearn.manifold import TSNE
import torch
import numpy as np


def Comp_SNE(input):
    input = input.cpu().detach().numpy()
    tsne = TSNE(n_components=2, init='pca', perplexity=10)
    output = tsne.fit_transform(input)

    print(input.shape, output.shape)
