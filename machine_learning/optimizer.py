import click 
import json 

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import torch.optim as optim 

from torch.utils.data import DataLoader
from machine_learning.producer import Producer 
from machine_learning.model import MLP 

from loguru import logger 

@click.command()
@click.option('--source', help='path to training data', type=click.Path(True), required=True)
@click.option('--batch_size', help='sier of batch during training', type=int, default=8, show_default=True)
@click.option('--nb_epochs', help='number of epochs for traing', type=int, default=64, show_default=True)
@click.option('--net_config', help='configuration of MLP', type=str, required=True)
@click.option('--store', help='path to model store', type=click.Path(False))
def learn(source, batch_size, nb_epochs, net_config, store):
	try:
		mlp_config = json.loads(net_config)
		mlp = MLP(**mlp_config)
		source = Producer(source)
		loader = DataLoader(dataset=source, batch_size=batch_size, shuffle=True)

		solver = optim.Adam(mlp.parameters(), lr=0.001)
		lossfn = nn.CrossEntropyLoss()

		message_fmt = 'Epoch : %03d | Error : %07.3f | Index : %04d'
		epoch_counter = 0 
		while epoch_counter < nb_epochs:
			index = 0
			for features, labels in loader: 
				output = mlp(features)
				error = lossfn(output, labels)
				solver.zero_grad()
				error.backward()
				solver.step()
				index = index + 1
				message_contents = (epoch_counter, error.item(), index) 
				print(message_fmt % message_contents)

			epoch_counter += 1

		th.save(mlp, store)
	except Exception as e: 
		logger.warning(e)

if __name__ == '__main__':
	print(' ... [learning] ... ')
	learn()