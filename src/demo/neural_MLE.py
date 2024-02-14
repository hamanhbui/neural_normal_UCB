from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import argparse
import pickle
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

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

class NeuralUCBDiag:
	def __init__(self, dim, lamdba=1, nu=1, hidden=100):
		self.n_arm = 4
		self.func = Network(dim, hidden_size=hidden).cuda()
		self.context_list = []
		self.reward = []
		self.lamdba = lamdba
		self.T = 1
		self.base_arm = torch.ones(self.n_arm) * 2

	def select(self, context):
		self.T += 1
		tensor = torch.from_numpy(context).float().cuda()
		output = self.func(tensor)
		mu, logsigma = output[:, 0], output[:, 1]
		
		UCB = torch.sqrt((np.log(self.T)/self.base_arm).cuda() *  torch.min(torch.ones(self.n_arm).cuda() * 1/4, torch.exp(logsigma)**2 + torch.sqrt((2*np.log(self.T))/self.base_arm).cuda()))
		sampled = mu + self.lamdba * UCB
		arm = np.argmax(sampled.cpu().detach().numpy())
		self.base_arm[arm] += 1
		return arm, logsigma, UCB

	def train(self, context, reward):
		self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
		self.reward.append(reward)
		optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
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

	def update_model(self, context, arm_select, reward):
		optimizer = optim.Adam(self.func.fc2.parameters())
		tensor = torch.from_numpy(context[arm_select]).float().cuda()
		optimizer.zero_grad()
		output = self.func(tensor)
		mu, logsigma = output[0], output[1]
		# loss = (reward[arm_select] - mu) ** 2
		loss = 2 * logsigma + ((reward[arm_select] - mu) / torch.exp(logsigma)) ** 2
		loss.backward()
		optimizer.step()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	with open ('contexts', 'rb') as fp:
		contexts = pickle.load(fp)

	with open ('rewards', 'rb') as fp:
		rewards = pickle.load(fp)

	parser.add_argument('--size', default=10000, type=int, help='bandit size')
	parser.add_argument('--dataset', default='mnist', metavar='DATASET')
	parser.add_argument('--shuffle', type=bool, default=1, metavar='1 / 0', help='shuffle the data set or not')
	parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None')
	parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
	parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')
	parser.add_argument('--hidden', type=int, default=100, help='network hidden size')

	args = parser.parse_args()
	l = NeuralUCBDiag(20, args.lamdba, args.nu, args.hidden)
	ucb_info = '_{:.3e}_{:.3e}'.format(args.lamdba, args.nu)

	regrets, list_logsigma, list_UCB = [], [], []
	summ = 0
	for t in range(10000):
		context, rwd = contexts[t], rewards[t]
		arm_select, log_sigma, UCB = l.select(context)
		r = rwd[arm_select]
		reg = np.max(rwd) - r
		summ+=reg
		list_logsigma.append(log_sigma)
		list_UCB.append(UCB)
		# l.update_model(context, arm_select, rwd)
		if t<2000:
			loss = l.train(context[arm_select], r)
		else:
			if t%100 == 0:
				loss = l.train(context[arm_select], r)
			# else:
			# 	l.update_model(context, arm_select, rwd)
		regrets.append(summ)
		if t % 100 == 0:
			print('{}: {:.3f}, {:.3e}'.format(t, summ, loss))
	   
	path = "out/logs/demo/neural_MLE"
	fr = open(path,'w')
	for i in regrets:
		fr.write(str(i))
		fr.write("\n")
	fr.close()

	with open('list_logsigma', 'wb') as fp:
		pickle.dump(list_logsigma, fp)

	with open('list_UCB', 'wb') as fp:
		pickle.dump(list_UCB, fp)