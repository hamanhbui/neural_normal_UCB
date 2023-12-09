import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import tensorflow as tf

def water_filling(n, a, sum_x=1):
	'''
	Boyd and Vandenberghe, Convex Optimization, example 5.2 page 145
	Water-filling.

	This problem arises in information theory, in allocating power to a set of
	n communication channels in order to maximise the total channel capacity.
	The variable x_i represents the transmitter power allocated to the ith channel,
	and log(α_i+x_i) gives the capacity or maximum communication rate of the channel.
	The objective is to minimise -∑log(α_i+x_i) subject to the constraint ∑x_i = 1
	'''

	# Declare variables and parameters
	x = cp.Variable(shape=n)
	alpha = cp.Parameter(n, nonneg=True)
	alpha.value = a

	# Choose objective function. Interpret as maximising the total communication rate of all the channels
	obj = cp.Maximize(cp.sum(cp.log(alpha + x)))

	# Declare constraints
	constraints = [x >= 0, cp.sum(x) - sum_x == 0]

	# Solve
	prob = cp.Problem(obj, constraints)
	prob.solve()
	if(prob.status=='optimal'):
		return prob.status, prob.value, x.value
	else:
		return prob.status, np.nan, np.nan

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

def greedy(X_mean, r_opt, X_mean_ood, r_opt_ood, K, Q, T, norm_eps, epsilon):
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
			if t < T/2:
				X_k = np.random.uniform(X_mean[i]-0.1, X_mean[i]+0.1)
			else:
				X_k = np.random.uniform(X_mean_ood[i]-0.1, X_mean_ood[i]+0.1)
			r_k = np.log(X_k + a[i] * norm_eps)
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
		# calculate regert
		if t < T/2:
			reg[t] = r_opt - r
		else:
			reg[t] = r_opt_ood - r
	
	return reg, reward

def non_stationary_greedy(X_mean, r_opt, X_mean_ood, r_opt_ood, K, Q, T, norm_eps, epsilon):
	reg = np.zeros(T)
	reward = np.zeros(T)
	mu_hat = np.zeros((K, Q+1)) # empirical mean
		
	for t in range(T):
		a = oracle(mu_hat, Q)
		if np.random.random() < epsilon:
			np.random.shuffle(a)
		# calculate the expected reward of action a
		r = 0
		for i in range(K):
			if t < T/2:
				X_k = np.random.uniform(X_mean[i]-0.1, X_mean[i]+0.1)
			else:
				X_k = np.random.uniform(X_mean_ood[i]-0.1, X_mean_ood[i]+0.1)
			r_k = np.log(X_k + a[i] * norm_eps)
			r += r_k
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / (t+1)
		reward[t] = r
		# calculate regert
		if t < T/2:
			reg[t] = r_opt - r
		else:
			reg[t] = r_opt_ood - r
	
	return reg, reward


def CUCB_RA(X_mean, r_opt, X_mean_ood, r_opt_ood, K, Q, T, norm_eps):
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
			if t < T/2:
				X_k = np.random.uniform(X_mean[i]-0.1, X_mean[i]+0.1)
			else:
				X_k = np.random.uniform(X_mean_ood[i]-0.1, X_mean_ood[i]+0.1)
			r_k = np.log(X_k + a[i] * norm_eps)
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
		# calculate regert
		if t < T/2:
			reg[t] = r_opt - r
		else:
			reg[t] = r_opt_ood - r
	
	return reg, reward

def CUCB_density_RA(X_mean, r_opt, X_mean_ood, r_opt_ood, K, Q, T, norm_eps):
	reg = np.zeros(T)
	reward = np.zeros(T)
	mu_hat = np.zeros((K, Q+1)) # empirical mean
	T_ka = np.ones((K, Q+1))# total number of times arm (k,a) is played
		
	for t in range(T):
		mu_bar = mu_hat + 0.1*np.sqrt(3*np.log(t+1)/(2*T_ka))
		if t >= T/2 and t <= T/2 + 5:
			mu_bar[3] = mu_hat[3] + 0.2*np.sqrt(3*np.log(t+1)/(2*T_ka[3]))
		a = oracle(mu_bar, Q)
		# calculate the expected reward of action a
		r = 0
		for i in range(K):
			if t < T/2:
				X_k = np.random.uniform(X_mean[i]-0.1, X_mean[i]+0.1)
			else:
				X_k = np.random.uniform(X_mean_ood[i]-0.1, X_mean_ood[i]+0.1)
			r_k = np.log(X_k + a[i] * norm_eps)
			r += r_k
			T_ka[i, a[i]] += 1
			mu_hat[i, a[i]] += (r_k - mu_hat[i, a[i]]) / T_ka[i, a[i]]
		reward[t] = r
		# calculate regert
		if t < T/2:
			reg[t] = r_opt - r
		else:
			reg[t] = r_opt_ood - r
	
	return reg, reward

# def grad_policy(X_mean, r_opt, K, Q, T, norm_eps):
# 	model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Dense(units=64, activation="relu"),
#             tf.keras.layers.Dense(units=K),
#         ]
#     )
# 	optimizer = tf.keras.optimizers.SGD()
# 	reg = np.zeros(T)
# 	for t in range(T):
# 		X = np.random.uniform(X_mean-0.1, X_mean+0.1)
# 		X = np.expand_dims(X, axis=0)
# 		with tf.GradientTape() as tape:
# 			mu_bar = model(X, training=True)
# 			a = oracle(mu_bar, Q)
# 			r = tf.reduce_sum(tf.math.log(X + a))
# 			# loss = (r_opt - r)**2
# 			loss = -r
# 		grads = tape.gradient(loss, model.trainable_weights)
# 		optimizer.apply_gradients(zip(grads, model.trainable_weights))

# 		# calculate regert
# 		reg[t] = r_opt - r
	
# 	return reg

def plot_by_normal(plt, value, label, color):
	mean = np.mean(np.array(value), axis = 0)
	std = np.std(np.array(value), axis = 0)
	# mean = mean[:5000]
	# std = std[:5000]
	plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3, color = color)
	plt.plot(mean, label = label, color = color)

if __name__ == "__main__":
	T = 10000
	N = 10
	Q = 5
	norm_eps = 1/Q
	K = 4
	reg_UCB, reg_greedy, eps_greedy, reg_gp = [], [], [], []
	reward_UCB, reward_greedy, reward_eps_greedy, reward_gp = [], [], [], []

	for i in range(N):
		X_mean = np.random.uniform(0.8, 1.2, K)
		stat, prob, x = water_filling(K, X_mean)
		print('Exp: {}'.format(i+1))
		print('Problem status: {}'.format(stat))
		print('Optimal communication rate = {:.4g} '.format(prob))
		print('Transmitter powers:\n{}'.format(x))
		X_mean_ood = np.array([X_mean[0], X_mean[1], X_mean[2], 10])
		stat_ood, prob_ood, x_ood = water_filling(K, X_mean_ood)
		reg, reward = CUCB_density_RA(X_mean, prob, X_mean_ood, prob_ood, K, Q, T, norm_eps)
		reg_gp.append(np.cumsum(reg))
		reward_gp.append(np.cumsum(reward))
		reg, reward = CUCB_RA(X_mean, prob, X_mean_ood, prob_ood, K, Q, T, norm_eps)
		reg_UCB.append(np.cumsum(reg))
		reward_UCB.append(np.cumsum(reward))
		reg, reward = greedy(X_mean, prob, X_mean_ood, prob_ood, K, Q, T, norm_eps, 0)
		reg_greedy.append(np.cumsum(reg))
		reward_greedy.append(np.cumsum(reward))
		reg, reward = greedy(X_mean, prob, X_mean_ood, prob_ood, K, Q, T, norm_eps, 0.1)
		eps_greedy.append(np.cumsum(reg))
		reward_eps_greedy.append(np.cumsum(reward))
	
	plot_by_normal(plt, reg_gp, "CUCB_density_RA", "#1f77b4")
	plot_by_normal(plt, reg_UCB, "CUCB_RA", "#ff7f0e")
	plot_by_normal(plt, reg_greedy, "Greedy", "#2ca02c")
	plot_by_normal(plt, eps_greedy, "$\epsilon$-Greedy $\epsilon=0.1$", "#d62728")
	
	plt.title("Online Water Filling [Boyd, Convex Optimization]")
	plt.xlabel("Steps")
	plt.ylabel("Cumulative Regret")
	plt.legend()
	plt.tight_layout()
	plt.savefig("out/regret_online_WF.png")
	
	plt.clf()

	plot_by_normal(plt, reward_gp, "CUCB_density_RA", "#1f77b4")
	plot_by_normal(plt, reward_UCB, "CUCB_RA", "#ff7f0e")
	plot_by_normal(plt, reward_greedy, "Greedy", "#2ca02c")
	plot_by_normal(plt, reward_eps_greedy, "$\epsilon$-Greedy $\epsilon=0.1$", "#d62728")
	
	plt.title("Online Water Filling [Boyd, Convex Optimization]")
	plt.xlabel("Steps")
	plt.ylabel("Cumulative Reward")
	plt.legend()
	plt.tight_layout()
	plt.savefig("out/reward_online_WF.png")

#Gradient Bandit Algorithms
#Neural bandit
#Gittins index