import numpy as np
import torch
from network import LaplacianMatrixRecorder

H = 128
W = 128
iterations = 500


def get_rand_matrix(H, W):
    return (torch.rand(H, W) > 0.5).int()


laplacian_graph = [get_rand_matrix(H, W)]
output_dir = 'output/20211009-lptest'

recorder = LaplacianMatrixRecorder(laplacian_graph, output_dir)
for it in range(iterations):
    new = [get_rand_matrix(H, W)]
    recorder.update(it, new)

recorder.plot('test.png')
