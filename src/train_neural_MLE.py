from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import argparse
import pickle
import os
import time
import random
import matplotlib.pyplot as plt

class Bandit_multi:
	def __init__(self, name, is_shuffle=True, seed=None):
		# Fetch data
		if name == 'mnist':
			X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
			# avoid nan, set nan as -1
			X[np.isnan(X)] = - 1
			X = normalize(X)
		elif name == 'covertype':
			X, y = fetch_openml('covertype', version=3, return_X_y=True)
			# avoid nan, set nan as -1
			X[np.isnan(X)] = - 1
			X = normalize(X)
		elif name == 'MagicTelescope':
			X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
			# avoid nan, set nan as -1
			X[np.isnan(X)] = - 1
			X = normalize(X)
		elif name == 'shuttle':
			X, y = fetch_openml('shuttle', version=1, return_X_y=True)
			# avoid nan, set nan as -1
			X[np.isnan(X)] = - 1
			X = normalize(X)
		else:
			raise RuntimeError('Dataset does not exist')
		# Shuffle data
		if is_shuffle:
			self.X, self.y = shuffle(X, y, random_state=seed)
		else:
			self.X, self.y = X, y
		# generate one_hot coding:
		self.y_arm = OrdinalEncoder(
			dtype=int).fit_transform(self.y.values.reshape((-1, 1)))
		# cursor and other variables
		self.cursor = 0
		self.size = self.y.shape[0]
		self.n_arm = np.max(self.y_arm) + 1
		self.dim = self.X.shape[1] * self.n_arm
		self.act_dim = self.X.shape[1]

	def step(self):
		assert self.cursor < self.size
		X = np.zeros((self.n_arm, self.dim))
		for a in range(self.n_arm):
			X[a, a * self.act_dim:a * self.act_dim +
				self.act_dim] = self.X[self.cursor]
		arm = self.y_arm[self.cursor][0]
		rwd = np.zeros((self.n_arm,))
		rwd[arm] = 1
		self.cursor += 1
		return X, rwd

	def finish(self):
		return self.cursor == self.size

	def reset(self):
		self.cursor = 0

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Network(nn.Module):
	def __init__(self, dim, hidden_size=100):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(dim, hidden_size)
		self.activate = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, 2)

	def forward(self, x):
		return self.fc2(self.activate(self.fc1(x)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='NeuralUCB')

	parser.add_argument('--size', default=15000, type=int, help='bandit size')
	parser.add_argument('--dataset', default='shuttle', metavar='DATASET')
	parser.add_argument('--shuffle', type=bool, default=1, metavar='1 / 0', help='shuffle the data set or not')
	parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None')
	parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
	parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')
	parser.add_argument('--hidden', type=int, default=128, help='network hidden size')

	args = parser.parse_args()
	use_seed = None if args.seed == 0 else args.seed
	b = Bandit_multi(args.dataset, is_shuffle=args.shuffle, seed=use_seed)
	bandit_info = '{}'.format(args.dataset)

	regrets = []
	summ = 0
	list_data = []
	for t in range(b.size):
		context, rwd = b.step()
		list_data.append((context, rwd))
	
	random.shuffle(list_data)
	train_set, val_set, test_set = list_data[:45000], list_data[45000:50000], list_data[50000:]

	train_set = []
	with open("context_list.pickle", "rb") as f:
		context_list = pickle.load(f)
	with open("reward.pickle", "rb") as f:
		reward = pickle.load(f)
	for idx in range(len(context_list)):
		train_set.append((context_list[idx], reward[idx]))

	train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)
	val_loader = DataLoader(val_set, batch_size = 1)
	test_loader = DataLoader(test_set, batch_size = 1)
	model =  Network(b.dim, hidden_size=args.hidden).cuda()
	# model.load_state_dict(torch.load("out.pth"))
	optimizer = optim.Adam(model.parameters())

	total_step = len(train_loader)
	train_loss, val_loss = [], []
	train_error, val_error = [], []
	n_epochs = 1000
	for epoch in range(n_epochs):
		print(f'Epoch {epoch}\n')
		batch_loss = 0
		batch_error = 0
		batch_idx = 0
		index = np.arange(len(context_list))
		for batch_idx, (samples, labels) in enumerate(train_loader):
			# if batch_idx < 10000000:
			# 	samples = samples.reshape(samples.shape[0] * samples.shape[1], samples.shape[2])
			# 	labels = labels.reshape(labels.shape[0] * labels.shape[1])
			# 	b = labels.nonzero()
			# 	samples = samples[b]
			# 	labels = labels[b]
			samples = samples.reshape(samples.shape[0] * samples.shape[1], samples.shape[2]).float().cuda()
			labels = labels.reshape(labels.shape[0], 1).cuda()

			optimizer.zero_grad()
			output = model(samples)
			#torch.Size([448, 1])
			mu, logsigma = output[:, 0], output[:, 1]
			mu = mu.reshape(mu.shape[0], 1)
			logsigma = logsigma.reshape(logsigma.shape[0], 1)

			loss = torch.mean(2 * logsigma + ((labels - mu) / torch.exp(logsigma)) ** 2)
			# loss = torch.mean((labels - mu) ** 2)
			batch_error += torch.mean((mu - labels)**2).item()
			
			loss.backward()
			optimizer.step()
			batch_loss += loss.item()

			if (batch_idx) % 20 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, n_epochs, batch_idx, total_step, loss.item()))

		train_loss.append(batch_loss/total_step)
		train_error.append(batch_error/total_step)
		print(f'\ntrain loss: {np.mean(train_loss):.4f}, train error: {np.mean(train_error):.4f}')

	plt.plot(train_error)
	plt.savefig("out/loss.png")
	torch.save(model.state_dict(), "out.pth")