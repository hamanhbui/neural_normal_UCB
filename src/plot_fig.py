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


linear_regret = get_regret('out/logs/linear_UCB')
neural_regret = get_regret('out/logs/neural_UCB')
linear_neural_regret = get_regret('out/logs/linear_neural_UCB')
ours_regret = get_regret('out/logs/neural_MLE')

plt.plot(linear_regret, label = "Linear_UCB")
plt.plot(neural_regret, label = "Neural_UCB")
plt.plot(linear_neural_regret, label = "Linear_Neural_UCB")
plt.plot(ours_regret, label = "Neural_MLE")
plt.legend()
plt.tight_layout()
plt.savefig("out/out.png")