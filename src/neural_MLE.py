from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import argparse
import pickle
import os
import time
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

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

class Network(nn.Module):
	def __init__(self, dim, hidden_size=100):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(dim, hidden_size)
		self.activate = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, 2)

	def forward(self, x):
		return self.fc2(self.activate(self.fc1(x)))

class NeuralUCBDiag:
	def __init__(self, dim, lamdba=1, nu=1, hidden=100):
		self.n_arm = 7
		self.func = Network(dim, hidden_size=hidden).cuda()
		# self.func.load_state_dict(torch.load("out.pth"))
		self.context_list = []
		self.reward = []
		self.lamdba = lamdba
		self.T = 0
		self.base_arm = torch.zeros(self.n_arm)

	def select(self, context):
		self.T += 1
		tensor = torch.from_numpy(context).float().cuda()
		output = self.func(tensor)
		mu, logsigma = output[:, 0], output[:, 1]

		# UCB = torch.sqrt((np.log(self.T)/self.base_arm).cuda() *  torch.min(torch.ones(self.n_arm).cuda() * 1/4, torch.exp(logsigma)**2 + torch.sqrt((2*np.log(self.T))/self.base_arm).cuda()))
		# UCB = torch.sqrt(16 * ((self.base_arm.cuda() * torch.exp(logsigma)**2).cuda()/(self.base_arm-1).cuda()) * (np.log(self.T - 1)/self.base_arm).cuda())
		UCB = torch.sqrt(16 * (torch.exp(logsigma)**2).cuda() * (np.log(self.T)/self.base_arm).cuda())
		# UCB = torch.exp(logsigma).cuda() 
		sampled = mu + self.lamdba * UCB
		arm = np.argmax(sampled.cpu().detach().numpy())
		self.base_arm[arm] += 1
		return arm, mu, logsigma, UCB

	def train(self, context, reward):
		# return 0
		self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
		self.reward.append(reward)
		optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
		# length = len(self.reward)
		# index = np.arange(length)
		# np.random.shuffle(index)
		# cnt = 0
		# tot_loss = 0
		train_set = []
		for idx in range(len(self.context_list)):
			train_set.append((self.context_list[idx], self.reward[idx]))
		train_loader = DataLoader(train_set, batch_size = 16, shuffle = True)
		total_step = len(train_loader)
		epoch = 0
		ite = 0
		while True:
			batch_loss = 0
			for batch_idx, (samples, labels) in enumerate(train_loader):
				samples = samples.reshape(samples.shape[0] * samples.shape[1], samples.shape[2]).float().cuda()
				labels = labels.reshape(labels.shape[0], 1).cuda()
				optimizer.zero_grad()
				output = self.func(samples.cuda())
				mu, logsigma = output[:, 0], output[:, 1]
				mu = mu.reshape(mu.shape[0], 1)
				logsigma = logsigma.reshape(logsigma.shape[0], 1)

				loss = torch.mean(2 * logsigma + ((labels - mu) / torch.exp(logsigma)) ** 2)
				
				loss.backward()
				optimizer.step()
				batch_loss += loss.item()
				ite += 1
				#HERE
				if ite >= 500:
					return batch_loss/total_step
			
		# while True:
		# 	batch_loss = 0
		# 	for idx in index:
		# 		c = self.context_list[idx]
		# 		r = self.reward[idx]
		# 		optimizer.zero_grad()

		# 		output = self.func(c.cuda())
		# 		mu, logsigma = output[:, 0], output[:, 1]
		# 		mu = mu.reshape(mu.shape[0], 1)
		# 		logsigma = logsigma.reshape(logsigma.shape[0], 1)

		# 		# logprob = -logsigma - 0.5*np.log(2*np.pi) - 0.5*((r-mu)/torch.exp(logsigma))**2
		# 		# loss = -logprob
		# 		if length >= 2000:
		# 			loss = torch.mean(2 * logsigma + ((r - mu) / torch.exp(logsigma)) ** 2)
		# 		else:
		# 			loss = torch.mean((r - mu) ** 2)

		# 		loss.backward()
		# 		optimizer.step()
		# 		batch_loss += loss.item()
		# 		tot_loss += loss.item()
		# 		cnt += 1
		# 		if cnt >= 1000:
		# 			return tot_loss / 1000
		# 	if batch_loss / length <= 1e-3:
		# 		return batch_loss / length

	def update_model(self, context, arm_select, reward):
		self.context_list.append(torch.from_numpy(context[arm_select].reshape(1, -1)).float())
		self.reward.append(reward[arm_select])

		# optimizer = optim.Adam(self.func.fc2.parameters())
		# tensor = torch.from_numpy(context[arm_select]).float().cuda()
		# optimizer.zero_grad()
		# output = self.func(tensor)
		# mu, logsigma = output[0], output[1]
		# # loss = (reward[arm_select] - mu) ** 2
		# loss = 2 * logsigma + ((reward[arm_select] - mu) / torch.exp(logsigma)) ** 2
		# loss.backward()
		# optimizer.step()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--size', default=15000, type=int, help='bandit size')
	parser.add_argument('--dataset', default='shuttle', metavar='DATASET')
	parser.add_argument('--shuffle', type=bool, default=0, metavar='1 / 0', help='shuffle the data set or not')
	parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None')
	parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
	parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')
	parser.add_argument('--hidden', type=int, default=100, help='network hidden size')

	args = parser.parse_args()
	use_seed = None if args.seed == 0 else args.seed
	b = Bandit_multi(args.dataset, is_shuffle=args.shuffle, seed=use_seed)
	bandit_info = '{}'.format(args.dataset)
	l = NeuralUCBDiag(b.dim, args.lamdba, args.nu, args.hidden)
	ucb_info = '_{:.3e}_{:.3e}_{}'.format(args.lamdba, args.nu, args.hidden)

	regrets, list_mu, list_logsigma, list_UCB = [], [], [], []
	summ = 0
	for t in range(min(args.size, b.size)):
		context, rwd = b.step()
		arm_select, mu, log_sigma, UCB = l.select(context)
		r = rwd[arm_select]
		reg = np.max(rwd) - r
		summ+=reg
		list_mu.append(mu)
		list_logsigma.append(log_sigma)
		list_UCB.append(UCB)
		# l.update_model(context, arm_select, rwd)
		if t<2000:
			loss = l.train(context[arm_select], r)
		else:
			if t%100 == 0:
				loss = l.train(context[arm_select], r)
			else:
				l.update_model(context, arm_select, rwd)
		regrets.append(summ)
		if t % 100 == 0:
			print('{}: {:.3f}, {:.3e}'.format(t, summ, loss))

	path = "out/logs/shuttle/neural_MLE"
	fr = open(path,'w')
	for i in regrets:
		fr.write(str(i))
		fr.write("\n")
	fr.close()

	with open('out/logs/shuttle/list_mu', 'wb') as fp:
		pickle.dump(list_mu, fp)

	with open('out/logs/shuttle/list_logsigma', 'wb') as fp:
		pickle.dump(list_logsigma, fp)

	with open('out/logs/shuttle/list_UCB', 'wb') as fp:
		pickle.dump(list_UCB, fp)
