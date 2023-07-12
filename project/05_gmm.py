from mlpr import *
from tqdm import tqdm
import random
import datetime
import svm
import gmm




pi_t_str = '\u03C0\u209C'  # U+209C SUBSCRIPT SMALL LETTER T
pi_tilde_str = '\u03C0\u0303'  # Combining U+0303 TILDE after U+03C0 PI

data, labels = load_data("data/Test.txt")
app = (0.5, 1, 1)
k = 5

num_males = data[:, labels == 0]
num_females = data[:, labels == 1]
pi_emp = round(num_females.shape[1] / num_males.shape[1], 3)
pi = pi_emp


# mean_mindcfs = []
# i = 1
# for i in tqdm(range(1, 4), disable=True):
#     mindcfs_to_be_averaged = []
#     for num_fold in list(range(1, k + 1)):
#         train_data, train_labels, test_data, test_labels = get_fold_data(data, labels, k, num_fold)
#         gmmc = gmm.GMM_classifier(i, 'full', 'untied')
#         gmmc.train(train_data, train_labels)
#         scores = gmmc.compute_scores(test_data)
#         mindcf = compute_min_DCF(scores, test_labels, app)
#
#         mindcfs_to_be_averaged.append(mindcf)
#     mean = round(np.array(mindcfs_to_be_averaged).mean(), 3)
#     print(f"{i}: {mean}")
#     mean_mindcfs.append(mean)
#
# print(mean_mindcfs)


# for i in tqdm(range(1, 4), disable=True):
mindcfs_to_be_averaged = []
for num_fold in tqdm(list(range(1, k + 1))):
    train_data, train_labels, test_data, test_labels = get_fold_data(data, labels, k, num_fold)
    # gmmc = gmm.GMM_classifier(2, 'full', 'tied')
    # gmmc = gmm.GMM_classifier(2, 'naive', 'untied')
    gmmc = gmm.GMM_classifier(2, 'naive', 'tied')
    gmmc.train(train_data, train_labels)
    scores = gmmc.compute_scores(test_data)
    mindcf = compute_min_DCF(scores, test_labels, app)

    mindcfs_to_be_averaged.append(mindcf)
mean = round(np.array(mindcfs_to_be_averaged).mean(), 3)
print(f"{2}: {mean}")

# print(mean_mindcfs)