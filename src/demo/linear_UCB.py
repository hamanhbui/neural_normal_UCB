from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import argparse
import pickle
import os
import time
import itertools

class LinearUCB:
	def __init__(self, dim, lamdba=1, nu=1):
		self.n_arm = 4
		self.theta = np.random.uniform(-1, 1, (self.n_arm, dim))
		self.b = np.zeros((self.n_arm, dim))
		self.A_inv = np.array([np.eye(dim) for _ in range(self.n_arm)])

	def select(self, context):
		ucb = np.array([np.sqrt(np.dot(context[a,:], np.dot(self.A_inv[a], context[a,:].T))) for a in range(self.n_arm)])
		mu = np.array([np.dot(context[a,:], self.theta[a]) for a in range(self.n_arm)])
		arm = np.argmax(mu + ucb)
		return arm

	def train(self, context, arm_select, reward):
		self.theta = np.array([np.matmul(self.A_inv[a], self.b[a]) for a in range(self.n_arm)])
		self.b[arm_select] += context[arm_select] * reward[arm_select]
		self.A_inv[arm_select] = inv_sherman_morrison(context[arm_select,:],self.A_inv[arm_select])

def inv_sherman_morrison(u, A_inv):
	"""Inverse of a matrix with rank 1 update.
	"""
	Au = np.dot(A_inv, u)
	A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
	return A_inv

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	with open ('contexts', 'rb') as fp:
		contexts = pickle.load(fp)

	with open ('rewards', 'rb') as fp:
		rewards = pickle.load(fp)

	with open ('psd_rewards', 'rb') as fp:
		psd_rewards = pickle.load(fp)

	parser.add_argument('--size', default=10000, type=int, help='bandit size')
	parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
	parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')

	args = parser.parse_args()
	l = LinearUCB(20, args.lamdba, args.nu)
	ucb_info = '_{:.3e}_{:.3e}'.format(args.lamdba, args.nu)

	regrets = []
	summ = 0
	for t in range(len(contexts)):
		context, rwd, psd_rwd = contexts[t], rewards[t], psd_rewards[t]
		arm_select = l.select(context)
		reg = np.max(psd_rwd) - psd_rwd[arm_select]
		summ+=reg
		l.train(context, arm_select, rwd)
		regrets.append(summ)
		if t % 100 == 0:
			print('{}: {:.3f}'.format(t, summ))
	   
	path = "out/logs/demo/linear_UCB"
	fr = open(path,'w')
	for i in regrets:
		fr.write(str(i))
		fr.write("\n")
	fr.close()