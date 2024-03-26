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


linear_regret = get_regret('out/logs/mnist/linear_UCB')
linear_regret2 = get_regret('out/logs/mnist/linear_UCB')
linear_regret3 = get_regret('out/logs/mnist/linear_UCB')
linear_regret = [linear_regret, linear_regret2, linear_regret3]

neural_regret = get_regret('out/logs/mnist/neural_UCB')
neural_regret2 = get_regret('out/logs/mnist/neural_UCB_1')
neural_regret3 = get_regret('out/logs/mnist/neural_UCB')
neural_regret = [neural_regret, neural_regret2, neural_regret3]

linear_neural_greedy_regret = get_regret('out/logs/mnist/linear_neural_greedy')
linear_neural_greedy_regret2 = get_regret('out/logs/mnist/linear_neural_greedy_1')
linear_neural_greedy_regret3 = get_regret('out/logs/mnist/linear_neural_greedy_2')
linear_neural_greedy_regret = [linear_neural_greedy_regret, linear_neural_greedy_regret2, linear_neural_greedy_regret3]

linear_neural_regret = get_regret('out/logs/mnist/linear_neural_UCB')
linear_neural_regret2 = get_regret('out/logs/mnist/linear_neural_UCB_1')
linear_neural_regret3 = get_regret('out/logs/mnist/linear_neural_UCB_2')
linear_neural_regret = [linear_neural_regret, linear_neural_regret2, linear_neural_regret3]

ours_regret_log = get_regret('out/logs/mnist/neural_MLE1')
ours_regret_log2 = get_regret('out/logs/mnist/neural_MLE2')
ours_regret_log3 = get_regret('out/logs/mnist/neural_MLE6')
ours_regret_log = [ours_regret_log, ours_regret_log2, ours_regret_log3]


def plot_by_normal(value, label, color = None):
	mean = np.mean(np.array(value), axis = 0)
	std = np.std(np.array(value), axis = 0)
	plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)
	if color is not None:
		plt.plot(mean, label = label, color = color)
	else:
		plt.plot(mean, label = label)

plot_by_normal(linear_regret, "Linear_UCB")
plot_by_normal(neural_regret, "Neural_MLE")
plot_by_normal(linear_neural_greedy_regret, "Linear_Neural_Greedy")
plot_by_normal(linear_neural_regret, "Linear_Neural_UCB")
plot_by_normal(ours_regret_log, "Neural_MLE", "blue")

plt.xlabel("Steps")
plt.ylabel("Cumulative regret")

plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("out/out.pdf")