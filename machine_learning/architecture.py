import operator as op 
import itertools as it, functools as ft 

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

class MLP(nn.Module):
    def __init__(self, layers, fun_names, drop_p=None):
        super(MLP, self).__init__()
        assert drop_p is None or len(drop_p) == len(fun_names)

        self.shapes = list(zip(layers[:-1], layers[1:]))
        self.linears = nn.ModuleList([ nn.Linear(m,n) for m,n in self.shapes ]) 
        self.activations = []
        for f_name in fun_names:
            if f_name != 'Softmax':
                fun = op.attrgetter(f_name)(nn)()
            else:
                fun = op.attrgetter(f_name)(nn)(dim=1)
            self.activations.append(fun)

        if drop_p is None:
            self.drops = [ nn.Identity() for _ in self.linears]
        else:
            self.drops = [ nn.Dropout(p) for p in drop_p ]

    def forward(self, X):
        reducer = lambda acc, crr: crr[2](crr[1](crr[0](acc)))
        data = zip(self.linears, self.activations, self.drops)
        out = ft.reduce(reducer, data, X)    
        return out 
    
    def project(self, X, depth): 
        self.eval()
        with th.no_grad():
            reducer = lambda acc, crr: crr[1](crr[0](acc))
            data = zip(self.linears[:depth], self.activations[:depth])
            out = ft.reduce(reducer, data, X)    
            return out 
    

