import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import tensorflow as tf

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
			r_k = -(1/(1-(list_data[t][i]/(a[i]+1)))) * (list_data[t][i]/sum(list_data[t]))
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
		# calculate regert
		reg[t] = list_r_opt[t] - r
	
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
			r_k = -(1/(1-(list_data[t][i]/(a[i]+1)))) * (list_data[t][i]/sum(list_data[t]))
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
		# calculate regert
		reg[t] = list_r_opt[t] - r
	
	return reg, reward

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
			tmp += -(1/(1-(list_data[j]/(list_out[i][j]+1)))) * (list_data[j]/sum(list_data))
		if tmp > max_tmp:
			max_tmp = tmp

	return max_tmp

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
		X_mean = np.array([0.2, 0.4, 0.6, 0.8])
		list_data, list_r_opt = [], []
		for t in range(T):
			list_X_k = []
			for j in range(K):
				list_X_k.append(np.random.uniform(X_mean[j]-0.1, X_mean[j]+0.1))
			r_opt = find_r_opt(list_X_k, Q)
                
			list_data.append(list_X_k)
			list_r_opt.append(r_opt)

		reg, reward = CUCB_RA(list_data, list_r_opt, K, Q, T, norm_eps)
		reg_UCB.append(np.cumsum(reg))
		reward_UCB.append(reward)
		reg, reward = greedy(list_data, list_r_opt, K, Q, T, norm_eps, 0)
		reg_greedy.append(np.cumsum(reg))
		reward_greedy.append(reward)
		reg, reward = greedy(list_data, list_r_opt, K, Q, T, norm_eps, 0.1)
		eps_greedy.append(np.cumsum(reg))
		reward_eps_greedy.append(reward)
	
	fig, axs = plt.subplots(1, 2, figsize=(11, 5))
	plot_by_normal(axs[0], reg_UCB, "CUCB_RA", "#ff7f0e")
	plot_by_normal(axs[0], reg_greedy, "Greedy", "#2ca02c")
	plot_by_normal(axs[0], eps_greedy, "$\epsilon$-Greedy $\epsilon=0.1$", "#d62728")
	
	axs[0].set_xlabel("Steps")
	axs[0].set_ylabel("Cumulative Regret")
	axs[0].legend()
	
	plot_by_normal(axs[1], reward_UCB, "CUCB_RA", "#ff7f0e")
	plot_by_normal(axs[1], reward_greedy, "Greedy", "#2ca02c")
	plot_by_normal(axs[1], reward_eps_greedy, "$\epsilon$-Greedy $\epsilon=0.1$", "#d62728")
	
	axs[1].set_xlabel("Steps")
	axs[1].set_ylabel("Reward")
	axs[1].legend()
	plt.tight_layout()
	plt.savefig("out/online_SA.pdf")

#Gradient Bandit Algorithms
#Neural bandit
#Gittins index