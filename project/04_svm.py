from mlpr import *
from tqdm import tqdm
import random
import datetime
import pickle
import time

# import svm

# def compute_min_DCF(scores, true_labels, cost_params):
#     return round(random.random(), 3)

pi_t_str = '\u03C0\u209C'  # U+209C SUBSCRIPT SMALL LETTER T
pi_tilde_str = '\u03C0\u0303'  # Combining U+0303 TILDE after U+03C0 PI

data, labels = load_data("data/Test.txt")
# z_data = z_norm(data)

mean_mindcfs = []
cost_list = list(10 ** i for i in range(-5, 3))
app = (0.5, 1, 1)
k = 5

num_males = data[:, labels == 0]
num_females = data[:, labels == 1]
pi_emp = round(num_females.shape[1] / num_males.shape[1], 3)
pi = pi_emp
print(f"pi = {pi}")


# bar = tqdm(total=2 * len(cost_list) * k)
# mindcfs_to_be_averaged = []
# print("linear")
# for idx, label in enumerate(["Linear", "Linear (z-norm)"]):
#     print(label)
#     mean_mindcfs = []
#     for c in tqdm(cost_list, disable=True):
#         mindcfs_to_be_averaged = []
#         for num_fold in list(range(1, k + 1)):
#             train_data, train_labels, test_data, test_labels = get_fold_data(data, labels, k, num_fold)
#             if idx == 1:
#                 train_data, test_data = znorm(train_data, test_data)
#             s = SupportVectorMachines(c, "linear", pi)
#             s.train(train_data, train_labels)
#             scores = s.compute_scores(test_data)
#             mindcf = compute_min_DCF(scores, test_labels, app)
#             mindcfs_to_be_averaged.append(mindcf)
#             bar.update(1)
#         mean = round(np.array(mindcfs_to_be_averaged).mean(), 3)
#         mean_mindcfs.append(mean)
#
#     print(mean_mindcfs)
# bar.close()

#
# print("poly2")
# for idx, label in enumerate(["Poly(2)", "Poly(2) (z-norm)"]):
#     bar = tqdm(total=len(cost_list) * k, desc=label)
#     mean_mindcfs = []
#     for c in tqdm(cost_list, disable=True):
#         mindcfs_to_be_averaged = []
#         for num_fold in list(range(1, k + 1)):
#             # print(".", end="")
#             train_data, train_labels, test_data, test_labels = get_fold_data(data, labels, k, num_fold)
#             if idx == 1:
#                 train_data, test_data = znorm(train_data, test_data)
#             s = SupportVectorMachines(c, "polynomial", pi, d=2)
#             s.train(train_data, train_labels)
#             scores = s.compute_scores(test_data)
#             mindcf = compute_min_DCF(scores, test_labels, app)
#             mindcfs_to_be_averaged.append(mindcf)
#             bar.update(1)
#         mean = round(np.array(mindcfs_to_be_averaged).mean(), 3)
#         mean_mindcfs.append(mean)
#     print()
#     print(mean_mindcfs)
#     file = open(label, 'wb')
#     pickle.dump(mean_mindcfs, file)
#     file.close()
#     bar.close()
#
#
# print("poly3")
# for idx, label in enumerate(["Poly(3)", "Poly(3) (z-norm)"]):
#     bar = tqdm(total=len(cost_list) * k, desc=label)
#     mean_mindcfs = []
#     for c in tqdm(cost_list, disable=True):
#         mindcfs_to_be_averaged = []
#         for num_fold in list(range(1, k + 1)):
#             train_data, train_labels, test_data, test_labels = get_fold_data(data, labels, k, num_fold)
#             if idx == 1:
#                 train_data, test_data = znorm(train_data, test_data)
#             s = SupportVectorMachines(c, "polynomial", pi, d=3)
#             s.train(train_data, train_labels)
#             scores = s.compute_scores(test_data)
#             mindcf = compute_min_DCF(scores, test_labels, app)
#             mindcfs_to_be_averaged.append(mindcf)
#             bar.update(1)
#         mean = round(np.array(mindcfs_to_be_averaged).mean(), 3)
#         mean_mindcfs.append(mean)
#     print()
#     print(mean_mindcfs)
#     file = open(label, 'wb')
#     pickle.dump(mean_mindcfs, file)
#     file.close()
#     bar.close()


print("rbf")
for g, label in [
    (10 ** -3, "SVM - RBF (log γ = −3)"),
    (10 ** -4, "SVM - RBF (log γ = −4)"),
    (10 ** -5, "SVM - RBF (log γ = −5)")
]:
    bar = tqdm(total=len(cost_list) * k, desc=label)
    mean_mindcfs = []
    for c in tqdm(cost_list, disable=True):
        mindcfs_to_be_averaged = []
        for num_fold in list(range(1, k + 1)):
            train_data, train_labels, test_data, test_labels = get_fold_data(data, labels, k, num_fold)
            s = SupportVectorMachines(c, "RBF", pi, gamma=g)
            s.train(train_data, train_labels)
            scores = s.compute_scores(test_data)
            mindcf = compute_min_DCF(scores, test_labels, app)
            mindcfs_to_be_averaged.append(mindcf)
            bar.update(1)
        mean = round(np.array(mindcfs_to_be_averaged).mean(), 3)
        mean_mindcfs.append(mean)
    bar.close()
    print()
    print(mean_mindcfs)
    file = open(label, 'wb')
    pickle.dump(mean_mindcfs, file)
    file.close()
    time.sleep(0.1)



# Plot plots
# check class rebalancing for best model(s)

# smv_z_norm = [0.275, 0.29, 0.28, 0.267, 0.255, 0.245, 0.236, 0.237]
# svm = [0.213, 0.167, 0.15, 0.142, 0.137, 0.139, 0.204, 0.254]
# plt.plot(cost_list, svm, label="svm")
# plt.plot(cost_list, smv_z_norm, label="svm (z-norm)")
# plt.title("Comparison between SVM and SVM (z-norm)")
# plt.xlabel('C')
# plt.ylabel(f"minDCF({pi_tilde_str}=0.5)")
# plt.xscale("log")
# plt.ylim(0, 1.0)
# plt.legend()
# timestamp = int(datetime.datetime.now().timestamp())
# plt.savefig('plots/svm_%d.png' % timestamp)
# plt.show()
