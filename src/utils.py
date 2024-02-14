import matplotlib.pyplot as plt
import numpy as np
def get_regret(filename):
	# Using readlines()
	file1 = open(filename, 'r')
	Lines = file1.readlines()

	regret_list = []
	count = 0
	# Strips the newline character
	for line in Lines:
		count += 1
		regret_list.append(float(line.strip()))
	
	return regret_list


linear_regret = get_regret('out/logs/demo/cos1/linear_UCB')
linear_regret2 = get_regret('out/logs/demo/cos2/linear_UCB')
linear_regret3 = get_regret('out/logs/demo/cos3/linear_UCB')
linear_regret = [linear_regret, linear_regret2, linear_regret3]

neural_regret = get_regret('out/logs/demo/cos1/neural_UCB')
neural_regret2 = get_regret('out/logs/demo/cos2/neural_UCB')
neural_regret3 = get_regret('out/logs/demo/cos3/neural_UCB')
neural_regret = [neural_regret, neural_regret2, neural_regret3]

linear_neural_regret = get_regret('out/logs/demo/cos1/linear_neural_UCB')
linear_neural_regret2 = get_regret('out/logs/demo/cos2/linear_neural_UCB')
linear_neural_regret3 = get_regret('out/logs/demo/cos3/linear_neural_UCB')
linear_neural_regret = [linear_neural_regret, linear_neural_regret2, linear_neural_regret3]

ours_regret_log = get_regret('out/logs/demo/cos1/neural_MLE')
ours_regret_log2 = get_regret('out/logs/demo/cos2/neural_MLE')
ours_regret_log3 = get_regret('out/logs/demo/cos3/neural_MLE')
ours_regret_log = [ours_regret_log, ours_regret_log2, ours_regret_log3]


def plot_by_normal(value, label):
	mean = np.mean(np.array(value), axis = 0)
	std = np.std(np.array(value), axis = 0)
	plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)
	plt.plot(mean, label = label)

plot_by_normal(linear_regret, "Linear_UCB")
plot_by_normal(neural_regret, "Neural_MLE")
plot_by_normal(linear_neural_regret, "Linear_Neural_UCB")
plot_by_normal(ours_regret_log, "Neural_MLE")

plt.xlabel("Steps")
plt.ylabel("Cumulative regret")

plt.legend()
plt.tight_layout()
plt.savefig("out/demo_cos.pdf")