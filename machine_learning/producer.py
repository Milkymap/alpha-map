import pickle 
import numpy as np 

import torch as th 
from torch.utils.data import Dataset 

class Producer(Dataset):
	def __init__(self, filepath):
		super(Producer, self).__init__()
		self.source = pickle.load(open(filepath, 'rb'))
		self.source = [ (m,n) for m,n in self.source if n in range(70, 80) ]
		
	def __len__(self):
		return len(self.source)

	def __getitem__(self, index):
		feature, label = self.source[index]
		return th.from_numpy(feature).float(), (label - 70) // 5 


