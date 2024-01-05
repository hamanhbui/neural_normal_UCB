import matplotlib.pyplot as plt

# Using readlines()
file1 = open('linear_UCB', 'r')
Lines = file1.readlines()

linear_regret = []
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    linear_regret.append(float(line.strip()))

# Using readlines()
file1 = open('neural_UCB', 'r')
Lines = file1.readlines()

neural_regret = []
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    neural_regret.append(float(line.strip()))

# Using readlines()
file1 = open('neural_MLE', 'r')
Lines = file1.readlines()

ours_regret = []
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    ours_regret.append(float(line.strip()))

plt.plot(linear_regret, label = "Linear_UCB")
plt.plot(neural_regret, label = "Neural_UCB")
plt.plot(ours_regret, label = "Neural_MLE")
plt.legend()
plt.tight_layout()
plt.savefig("out.png")