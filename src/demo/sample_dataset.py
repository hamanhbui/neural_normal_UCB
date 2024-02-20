import numpy as np
import pickle
import itertools

class Bandit_multi:
	def __init__(self):
		self.dim = 20
		self.n_arms = 4
		a = np.random.randn(self.dim)
		a /= np.linalg.norm(a, ord=2)
		# A = 1.0 * np.random.randn(self.dim, self.dim)
		# self.h = lambda x: 10*(np.dot(a, x))**2
		# self.h = lambda x: np.matmul(np.matmul(np.matmul(x, A), A), x)
		self.h = lambda x: np.cos(3*np.dot(x, a))
		self.noise_std = 1.0
		self.size = 10000

	def step(self):
		x = np.random.randn(self.n_arms, self.dim)
		x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.dim).reshape(self.n_arms, self.dim)
		
		noise = self.noise_std*np.random.randn()
		rwd = np.array(
			[
				self.h(x[k]) + noise
				for k in itertools.product(range(self.n_arms))
			]
		).reshape(self.n_arms)
		
		return x, rwd


b = Bandit_multi()
contexts, rewards = [], []
for t in range(b.size):
    context, rwd = b.step()
    contexts.append(context)
    rewards.append(rwd)

with open('contexts', 'wb') as fp:
    pickle.dump(contexts, fp)

with open('rewards', 'wb') as fp:
    pickle.dump(rewards, fp)
