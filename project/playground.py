from mlpr import *
import pickle
import gmm
from tqdm import tqdm
import time

# bar = tqdm(total = 9)
# for i in range(10):
#     time.sleep(5)
#     bar.update(1)
# bar.close()
# print("hello")
v = "hi"
file = open("test(test)", 'wb')
pickle.dump(v, file)
file.close()

# data, labels = load_data("data/Test.txt")
#
# mean, cov = estimate_mean_and_covariance(data)
# gmm_calculated = [(1, mean, cov)]
#
# gmmc = gmm.GMM_classifier(3, 'full', 'untied')
#
# gmmc.train(data, labels)
# print(len(gmmc.gmm0))
# scores = gmmc.compute_scores(data)
# print(compute_min_DCF(scores, labels, (0.5, 1, 1)))
# print(data[:, labels==1].shape[1])
# print(data[:, labels==0].shape[1])
#
# train_data, train_labels, test_data, test_labels = get_fold_data(data, labels, 3, 1)
# scores = weighted_logistic_reg_score(train_data, train_labels, test_data, 1e-4, 0.5)
# predictions = (scores > 0)*1
#
# print(confusion_matrix_binary(predictions, test_labels))