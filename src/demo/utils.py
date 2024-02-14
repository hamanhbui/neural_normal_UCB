import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

with open ('out/logs/mnist/list_logsigma', 'rb') as fp:
	list_logsigma = pickle.load(fp)

for i in range(len(list_logsigma)):
    list_logsigma[i] = torch.exp(list_logsigma[i])
    list_logsigma[i] = list_logsigma[i].cpu().detach().numpy()

list_logsigma = np.array(list_logsigma)
fig, ax = plt.subplots()
plt.hist(list_logsigma[:,0], density=True, alpha=0.8, bins = 100, label = "arm 1")
plt.hist(list_logsigma[:,1], density=True, alpha=0.8, bins = 100, label = "arm 2")
plt.hist(list_logsigma[:,2], density=True, alpha=0.8, bins = 100, label = "arm 3")
plt.hist(list_logsigma[:,3], density=True, alpha=0.8, bins = 100, label = "arm 4")
plt.hist(list_logsigma[:,4], density=True, alpha=0.8, bins = 100, label = "arm 5")
plt.hist(list_logsigma[:,5], density=True, alpha=0.8, bins = 100, label = "arm 6")
plt.hist(list_logsigma[:,6], density=True, alpha=0.8, bins = 100, label = "arm 7")
plt.hist(list_logsigma[:,7], density=True, alpha=0.8, bins = 100, label = "arm 8")
plt.hist(list_logsigma[:,8], density=True, alpha=0.8, bins = 100, label = "arm 9")
plt.hist(list_logsigma[:,9], density=True, alpha=0.8, bins = 100, label = "arm 10")
# print(np.min(list_logsigma[:,3]))
# print(np.max(list_logsigma[:,3]))
# plt.plot(list_logsigma[:,0], label = "arm 1")
# plt.plot(list_logsigma[:,1], label = "arm 2")
# plt.plot(list_logsigma[:,2], label = "arm 3")
# plt.plot(list_logsigma[:,3], label = "arm 4")
# plt.plot(list_logsigma[:,4], label = "arm 5")
# plt.plot(list_logsigma[:,5], label = "arm 6")
# plt.plot(list_logsigma[:,6], label = "arm 7")
# plt.plot(list_logsigma[:,7], label = "arm 8")
# plt.plot(list_logsigma[:,8], label = "arm 9")
# plt.plot(list_logsigma[:,9], label = "arm 10")
# ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
plt.xlabel("Steps")
plt.ylabel("UCB")
plt.legend()
plt.tight_layout()
plt.savefig("out.png")