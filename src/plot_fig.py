import matplotlib.pyplot as plt

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


linear_regret = get_regret('out/logs/mnist2/linear_UCB')
neural_regret = get_regret('out/logs/mnist2/neural_UCB')
linear_neural_greedy_regret = get_regret('out/logs/mnist2/linear_neural_greedy')
linear_neural_regret = get_regret('out/logs/mnist2/linear_neural_UCB')
ours_regret_log = get_regret('out/logs/mnist2/neural_MLE')

plt.plot(linear_regret, label = "Linear_UCB")
plt.plot(neural_regret, label = "Neural_UCB")
plt.plot(linear_neural_greedy_regret, label = "Linear_Neural_greedy")
plt.plot(linear_neural_regret, label = "Linear_Neural_UCB")
plt.plot(ours_regret_log, label = "Neural_MLE", color = "blue")
plt.xlabel("Steps")
plt.ylabel("Cumulative regret")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("out/out.png")