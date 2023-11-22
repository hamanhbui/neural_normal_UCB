import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe1 = pd.read_excel('data/texas_ICU_beds.xlsx')

list_data = []
for index in range(2, dataframe1.shape[1], 1):     
	columnSeriesObj = dataframe1.iloc[:, index]
	list_data.append(columnSeriesObj.values)
	
list_data = np.array(list_data)

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

def greedy(X_mean, K, Q, T, epsilon):
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
			r_k = -np.abs(X_k - a[i])
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
	
	return reward

def CUCB_RA(X_mean, K, Q, T):
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
			r_k = -np.abs(X_k - a[i])
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
	
	return reward

def plot_by_normal(plt, value, label, color):
	mean = np.mean(np.array(value), axis = 0)
	std = np.std(np.array(value), axis = 0)
	plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3, color = color)
	plt.plot(mean, label = label, color = color)

if __name__ == "__main__":
	reg_UCB, reg_greedy, eps_greedy, reg_gp = [], [], [], []
	reward_UCB, reward_greedy, reward_eps_greedy, reward_gp = [], [], [], []

	N = 10
	Q = 1000
	T = list_data.shape[0]
	K = list_data.shape[1]
	reward_UCB, reward_greedy, reward_eps_greedy, reward_gp = [], [], [], []

	for i in range(N):
		reward = CUCB_RA(list_data, K, Q, T)
		reward_UCB.append(reward)
		reward = greedy(list_data, K, Q, T, 0)
		reward_greedy.append(reward)
		reward = greedy(list_data, K, Q, T, 0.1)
		eps_greedy.append(reward)

	plot_by_normal(plt, reward_UCB, "CUCB_RA", "#ff7f0e")
	plot_by_normal(plt, reward_greedy, "Greedy", "#2ca02c")
	plot_by_normal(plt, eps_greedy, "$\epsilon$-Greedy $\epsilon=0.1$", "#d62728")
	
	plt.title("Patient Allocation")
	plt.xlabel("Steps")
	plt.ylabel("Reward")
	plt.legend()
	plt.tight_layout()
	plt.savefig("out/reward_hospital_RA.png")