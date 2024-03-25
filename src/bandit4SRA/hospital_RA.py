import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataframe = pd.read_csv('data/covid_data2/out.csv')
list_tmp = []
for index in range(1, dataframe.shape[1], 1): 
	for t in range(1):    
		columnSeriesObj = dataframe.iloc[:, index]
		list_tmp.append(columnSeriesObj.values)

list_data = []
for i in range(len(list_tmp)):
	list_data.append([])
	for j in range(len(list_tmp[i])):
		tmp = list_tmp[i][j].strip('][').split(', ')
		list_data[i].append(tmp)

list_data = np.array(list_data, dtype=float)
list_tmp = list_data.reshape(list_data.shape[0], -1)
mean = np.mean(list_tmp, axis=0)
cov = np.cov(list_tmp, rowvar=0)

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

def greedy(list_data, K, list_Q, T, epsilon):
	reg = np.zeros(T)
	reward = np.zeros(T)
	mu_hat = np.zeros((K, max(list_Q)+1)) # empirical mean
	T_ka = np.zeros((K, max(list_Q)+1))# total number of times arm (k,a) is played
		
	for t in range(T):
		a = oracle(mu_hat, list_Q[t])
		if np.random.random() < epsilon:
			np.random.shuffle(a)
		# calculate the expected reward of action a
		r = 0
		for i in range(K):
			X_k = list_data[t][i][6]
			r_k = -max(0, a[i] - X_k)
			# r_k = -np.abs(X_k - a[i])
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
		# calculate regert
		r_opt = -max(0, list_Q[t] - sum(list_data[t][:,6]))
		reg[t] = r_opt - r
	
	return reg, reward

def CUCB_RA(list_data, K, list_Q, T):
	reg = np.zeros(T)
	reward = np.zeros(T)
	mu_hat = np.zeros((K, max(list_Q)+1)) # empirical mean
	T_ka = np.ones((K, max(list_Q)+1))# total number of times arm (k,a) is played
		
	for t in range(T):
		mu_bar = mu_hat + 0.1*np.sqrt(3*np.log(t+1)/(2*T_ka))
		a = oracle(mu_bar, list_Q[t])
		# calculate the expected reward of action a
		r = 0
		for i in range(K):
			X_k = list_data[t][i][6]
			r_k = -max(0, a[i] - X_k)
			# r_k = -np.abs(X_k - a[i])
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
		# calculate regert
		r_opt = -max(0, list_Q[t] - sum(list_data[t][:,6]))
		reg[t] = r_opt - r
	
	return reg, reward

def Density_CUCB_RA(list_data, K, list_Q, T):
	reg = np.zeros(T)
	reward = np.zeros(T)
	mu_hat = np.zeros((K, max(list_Q)+1)) # empirical mean
	T_ka = np.ones((K, max(list_Q)+1))# total number of times arm (k,a) is played
		
	for t in range(T):
		if t >= T/2 and t <= T/2 + 5:
			mu_bar = mu_hat + 1*np.sqrt(3*np.log(t+1)/(2*T_ka))
		else:
			mu_bar = mu_hat + 0.1*np.sqrt(3*np.log(t+1)/(2*T_ka))
		a = oracle(mu_bar, list_Q[t])
		# calculate the expected reward of action a
		r = 0
		for i in range(K):
			X_k = list_data[t][i][6]
			r_k = -max(0, a[i] - X_k)
			# r_k = -np.abs(X_k - a[i])
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
		# calculate regert
		r_opt = -max(0, list_Q[t] - sum(list_data[t][:,6]))
		reg[t] = r_opt - r
	
	return reg, reward

def CNeural_RA(list_data, K, list_Q, T):
	reg = np.zeros(T)
	reward = np.zeros(T)
	T_ka = np.ones((K, max(list_Q)+1))# total number of times arm (k,a) is played
	metrics = {'nll': tf.keras.metrics.Mean()}
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(64, activation="relu"),
		tf.keras.layers.Dense(64, activation="relu"),
		tf.keras.layers.Dense(64, activation="relu"),
		tf.keras.layers.Dense(max(list_Q)+1),
	])
	optimizer = tf.keras.optimizers.Adam()
	x_train, a_train, y_train = [], [], []
	train_loss = np.zeros(T)
	for t in range(T):
		context = list_data[t][:, :-1]
		# context = np.expand_dims(list_data[t][:, 4], axis = 1)
		mu_hat = model(context)
		mu_bar = mu_hat + 0.1*np.sqrt(3*np.log(t+1)/(2*T_ka))
		a = oracle(mu_bar, list_Q[t])
		r = []
		for i in range(K):
			X_k = list_data[t][i][6]
			r_k = -max(0, a[i] - X_k)
			r.append(r_k)
			T_ka[i, a[i]] += 1
		reward[t] = sum(r)
		# calculate regert
		r_opt = -max(0, list_Q[t] - sum(list_data[t][:,6]))
		reg[t] = r_opt - sum(r)

		#Update model
		# x_train, a_train, y_train = [], [], []
		a_train.append(a)
		x_train.append(context)
		y_train.append(np.array(r, dtype = np.float32))
		train_dataset = tf.data.Dataset.from_tensor_slices((x_train, a_train, y_train))
		# train_dataset = train_dataset.shuffle(32).batch(32)

		for step, (x_batch_train, a_batch_train, y_batch_train) in enumerate(train_dataset):
			with tf.GradientTape() as tape:
				mu_bar = model(x_batch_train)
				# a = oracle(mu_bar, list_Q[t])
				mu_bar = tf.gather(mu_bar, indices = a_batch_train, batch_dims=1)
				loss = tf.reduce_sum(tf.pow(mu_bar - y_batch_train, 2))
				
			grads = tape.gradient(loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
			metrics['nll'].update_state(loss)

		train_loss[t] = metrics['nll'].result()
		for metric in metrics.values():
			metric.reset_states()
		
		print(t)

	plt.plot(train_loss)
	plt.savefig("out/train_loss.png")
	plt.clf()
	return reg, reward


def plot_by_normal(plt, value, label, color):
	mean = np.mean(np.array(value), axis = 0)
	std = np.std(np.array(value), axis = 0)
	plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3, color = color)
	plt.plot(mean, label = label, color = color)

def get_list_Q(list_data):
	list_Q = []
	for i in range(len(list_data)):
		list_Q.append(int(sum(list_data[i][:,5])))
	return np.array(list_Q)

if __name__ == "__main__":
	N = 1
	T = list_data.shape[0]
	K = list_data.shape[1]
	reg_UCB, reg_greedy, eps_greedy, reg_gp = [], [], [], []
	reward_UCB, reward_greedy, reward_eps_greedy, reward_gp = [], [], [], []
	# list_Q = get_list_Q(list_data)

	for i in range(N):
		list_Q = np.random.normal(2000, 100, T)
		list_Q = list_Q.astype(int)
		reg, reward = CNeural_RA(list_data, K, list_Q, T)
		reg_gp.append(np.cumsum(reg))
		reward_gp.append(reward)
		reg, reward = CUCB_RA(list_data, K, list_Q, T)
		reg_UCB.append(np.cumsum(reg))
		reward_UCB.append(reward)
		reg, reward = greedy(list_data, K, list_Q, T, 0)
		reg_greedy.append(np.cumsum(reg))
		reward_greedy.append(reward)
		reg, reward = greedy(list_data, K, list_Q, T, 0.1)
		eps_greedy.append(np.cumsum(reg))
		reward_eps_greedy.append(reward)

	fig, axs = plt.subplots(1, 2, figsize=(11, 5))
	plot_by_normal(axs[0], reg_UCB, "CUCB_RA", "#ff7f0e")
	plot_by_normal(axs[0], reg_greedy, "Greedy", "#2ca02c")
	plot_by_normal(axs[0], eps_greedy, "$\epsilon$-Greedy $\epsilon=0.1$", "#d62728")
	plot_by_normal(axs[0], reg_gp, "CNeural_RA", "#1f77b4")

	axs[0].set_xlabel("Steps")
	axs[0].set_ylabel("Cumulative Regret")
	axs[0].legend()

	plot_by_normal(axs[1], reward_UCB, "CUCB_RA", "#ff7f0e")
	plot_by_normal(axs[1], reward_greedy, "Greedy", "#2ca02c")
	plot_by_normal(axs[1], reward_eps_greedy, "$\epsilon$-Greedy $\epsilon=0.1$", "#d62728")
	plot_by_normal(axs[1], reward_gp, "CNeural_RA", "#1f77b4")
	
	axs[1].set_xlabel("Steps")
	axs[1].set_ylabel("Reward")
	axs[1].legend()
	plt.tight_layout()
	plt.savefig("out/hospital_RA2.png")