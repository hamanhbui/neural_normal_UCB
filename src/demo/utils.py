import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

with open ('out/logs/shuttle/list_mu', 'rb') as fp:
	list_mu = pickle.load(fp)

with open ('out/logs/shuttle/list_UCB', 'rb') as fp:
	list_UCB = pickle.load(fp)

# for i in range(len(list_logsigma)):
#     list_logsigma[i] = torch.exp(list_logsigma[i])
#     list_logsigma[i] = list_logsigma[i].cpu().detach().numpy()

list_out = []
base_arm = torch.ones(10) * 2
for i in range(len(list_mu)):
    sampled = list_mu[i] + list_UCB[i] * 0.01
    arm = np.argmax(sampled.cpu().detach().numpy())
    base_arm[arm] += 1
    list_out.append(base_arm.clone().numpy())

# list_out = []
# for i in range(len(list_UCB)):
#     # sampled = list_mu[i] + list_UCB[i] * 0.01
#     # arm = np.argmax(sampled.cpu().detach().numpy())
#     list_out.append(list_UCB[i].cpu().detach().numpy())

list_out = np.array(list_out)
fig, ax = plt.subplots()
# plt.hist(list_out[:,0], density=True, alpha=0.8, bins = 100, label = "arm 1")
# plt.hist(list_out[:,1], density=True, alpha=0.8, bins = 100, label = "arm 2")
# plt.hist(list_out[:,2], density=True, alpha=0.8, bins = 100, label = "arm 3")
# plt.hist(list_out[:,3], density=True, alpha=0.8, bins = 100, label = "arm 4")
# plt.hist(list_out[:,4], density=True, alpha=0.8, bins = 100, label = "arm 5")
# plt.hist(list_out[:,5], density=True, alpha=0.8, bins = 100, label = "arm 6")
# plt.hist(list_out[:,6], density=True, alpha=0.8, bins = 100, label = "arm 7")
# plt.hist(list_out[:,7], density=True, alpha=0.8, bins = 100, label = "arm 8")
# plt.hist(list_out[:,8], density=True, alpha=0.8, bins = 100, label = "arm 9")
# plt.hist(list_out[:,9], density=True, alpha=0.8, bins = 100, label = "arm 10")
# print(np.min(list_out[:,3]))
# print(np.max(list_out[:,3]))
plt.plot(list_out[:,0], label = "arm 1")
plt.plot(list_out[:,1], label = "arm 2")
plt.plot(list_out[:,2], label = "arm 3")
plt.plot(list_out[:,3], label = "arm 4")
plt.plot(list_out[:,4], label = "arm 5")
plt.plot(list_out[:,5], label = "arm 6")
plt.plot(list_out[:,6], label = "arm 7")
# plt.plot(list_out[:,7], label = "arm 8")
# plt.plot(list_out[:,8], label = "arm 9")
# plt.plot(list_out[:,9], label = "arm 10")
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
plt.xlabel("Steps")
plt.ylabel("UCB")
plt.legend()
plt.tight_layout()
plt.savefig("out/out.png")