import os
import random
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_data(file_name):
	dataframe = pd.read_csv(file_name)
	list_AP = {"Bldg3AP91": 0, "Bldg3AP92": 0, "Bldg3AP94": 0, "Bldg3AP95": 0}
	a = 4
	list_out = []
	for i in range(len(dataframe)):
		list_AP[dataframe.loc[i, 'AP']] += 1
		if i != len(dataframe)-1:
			str_1 = dataframe.loc[i, 'timestamp'].split(':')[0] + ":" + dataframe.loc[i, 'timestamp'].split(':')[1]
			str_2 = dataframe.loc[i+1, 'timestamp'].split(':')[0] + ":" + dataframe.loc[i+1, 'timestamp'].split(':')[1]
			if str_1 != str_2:
				if list_AP["Bldg3AP91"] > a or list_AP["Bldg3AP92"] > a or list_AP["Bldg3AP94"] > a or list_AP["Bldg3AP95"] > a:
					list_out.append([list_AP["Bldg3AP91"], list_AP["Bldg3AP92"], list_AP["Bldg3AP94"], list_AP["Bldg3AP95"]])
				list_AP = {"Bldg3AP91": 0, "Bldg3AP92": 0, "Bldg3AP94": 0, "Bldg3AP95": 0}
		else:
			list_out.append([list_AP["Bldg3AP91"], list_AP["Bldg3AP92"], list_AP["Bldg3AP94"], list_AP["Bldg3AP95"]])

	return np.array(list_out)

list_data = get_data('data/dynamic_user_RA/out.csv')

def oracle(mu, Q):
	a = np.zeros(mu.shape[0], dtype=int) # action
	# marginal gain of resources
	tmp = np.array(mu[:,1] - mu[:,0])
	for i in range(Q):
		j = np.argmax(tmp)
		a[j] += 1
		if a[j] < Q:
			tmp[j] = mu[j, a[j]+1] - mu[j, a[j]]

	return a

def find_r_opt(list_data, Q):
	def sums(length, total_sum):
		if length == 1:
			yield (total_sum,)
		else:
			for value in range(total_sum + 1):
				for permutation in sums(length - 1, total_sum - value):
					yield (value,) + permutation

	list_out = list(sums(len(list_data), Q))
	max_tmp = -np.inf
	for i in range(len(list_out)):
		tmp = 0
		for j in range(len(list_data)):
			tmp += 0.2*(list_data[j] + list_out[i][j]) * np.exp(-0.2*(list_data[j] + list_out[i][j]))
		if tmp > max_tmp:
			max_tmp = tmp

	return max_tmp

def greedy(list_data, list_r_opt, K, Q, T, norm_eps, epsilon):
	reg = np.zeros(T)
	reward = np.zeros(T)
	mu_hat = np.zeros((K, Q+1)) # empirical mean
	T_ka = np.zeros((K, Q+1))# total number of times arm (k,a) is played
	for t in range(T):
		a = oracle(mu_hat, Q)
		if np.random.random() < epsilon:
			np.random.shuffle(a)
		# calculate the expected reward of action a
		r = 0
		for i in range(K):
			X_k = list_data[t][i]
			r_k = 0.2*(X_k + a[i]) * np.exp(-0.2*(X_k + a[i]))
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
		# calculate regert
		r_opt = list_r_opt[t]
		reg[t] = r_opt - r
	
	return reg, reward

def CUCB_RA(list_data, list_r_opt, K, Q, T, norm_eps):
	reg = np.zeros(T)
	reward = np.zeros(T)
	mu_hat = np.zeros((K, Q+1)) # empirical mean
	T_ka = np.ones((K, Q+1))# total number of times arm (k,a) is played
	for t in range(T):
		mu_bar = mu_hat + 0.1*np.sqrt(3*np.log(t+1)/(2*T_ka))
		a = oracle(mu_bar, Q)
		# calculate the expected reward of action a
		r = 0
		for i in range(K):
			X_k = list_data[t][i]
			r_k = 0.2*(X_k + a[i]) * np.exp(-0.2*(X_k + a[i]))
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
		# calculate regert
		r_opt = list_r_opt[t]
		reg[t] = r_opt - r
	
	return reg, reward


def plot_by_normal(plt, value, label, color):
	mean = np.mean(np.array(value), axis = 0)
	std = np.std(np.array(value), axis = 0)
	plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3, color = color)
	plt.plot(mean, label = label, color = color)

if __name__ == "__main__":
	T = 2000
	N = 10
	Q = 8
	norm_eps = 1/Q
	K = 4
	reg_UCB, reg_greedy, eps_greedy, reg_gp = [], [], [], []
	reward_UCB, reward_greedy, reward_eps_greedy, reward_gp = [], [], [], []
	
	for i in range(N):
		list_tmp, list_r_opt = [], []
		for i in range(T):
			tmp = random.choice(list_data)
			r_opt = find_r_opt(tmp, Q)
			list_tmp.append(tmp)
			list_r_opt.append(r_opt)
		list_data = list_tmp

		reg, reward = CUCB_RA(list_data, list_r_opt, K, Q, T, norm_eps)
		reg_UCB.append(np.cumsum(reg))
		reward_UCB.append(np.cumsum(reward))
		reg, reward = greedy(list_data, list_r_opt, K, Q, T, norm_eps, 0)
		reg_greedy.append(np.cumsum(reg))
		reward_greedy.append(np.cumsum(reward))
		reg, reward = greedy(list_data, list_r_opt, K, Q, T, norm_eps, 0.1)
		eps_greedy.append(np.cumsum(reg))
		reward_eps_greedy.append(np.cumsum(reward))
	
	plot_by_normal(plt, reg_UCB, "CUCB_RA", "#ff7f0e")
	plot_by_normal(plt, reg_greedy, "Greedy", "#2ca02c")
	plot_by_normal(plt, eps_greedy, "$\epsilon$-Greedy $\epsilon=0.1$", "#d62728")
	
	plt.title("Online Water Filling [Boyd, Convex Optimization]")
	plt.xlabel("Steps")
	plt.ylabel("Cumulative Regret")
	plt.legend()
	plt.tight_layout()
	plt.savefig("out/regret_dynamic_user_RA.png")
	
	plt.clf()

	plot_by_normal(plt, reward_UCB, "CUCB_RA", "#ff7f0e")
	plot_by_normal(plt, reward_greedy, "Greedy", "#2ca02c")
	plot_by_normal(plt, reward_eps_greedy, "$\epsilon$-Greedy $\epsilon=0.1$", "#d62728")
	
	plt.title("Online Water Filling [Boyd, Convex Optimization]")
	plt.xlabel("Steps")
	plt.ylabel("Cumulative Reward")
	plt.legend()
	plt.tight_layout()
	plt.savefig("out/reward_dynamic_user_RA.png")