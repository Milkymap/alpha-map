import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 
import operator as op 
import itertools as it, functools as ft 

class MLP(nn.Module):
	def __init__(self, layers, activations, drop_p):
		super(MLP, self).__init__()
		assert len(drop_p) == len(activations)
		assert len(activations) == len(layers) - 1 
		self.shapes = list(zip(layers[:-1], layers[1:]))
		self.linears = nn.ModuleList([ nn.Linear(m, n) for m,n in self.shapes ])
		self.activations = [ op.attrgetter(fn)(nn)() for fn in activations]
		self.dropouts = [ nn.Dropout(p) for p in drop_p ]

	def forward(self, X): 
		reducer = lambda acc, crr: ft.reduce(lambda a,f: f(a), crr, acc) 
		iterable = list(zip(self.linears, self.activations, self.dropouts))
		return ft.reduce(reducer, iterable, X)

if __name__ == '__main__':
	print(' ... [modelization] ... ')
	mlp = MLP(
		layers=[4096, 256, 128, 10], 
		activations=['ReLU', 'ReLU', 'Identity'], 
		drop_p=[0.1, 0.2, 0.0]
	)

	print(mlp)