from mlpr import *

data, labels = load_data("data/Test.txt")

train_data, train_labels, test_data, test_labels = get_fold_data(data, labels, 3, 1)
scores = weighted_logistic_reg_score(train_data, train_labels, test_data, 1e-4, 0.5)
predictions = (scores > 0)*1

print(confusion_matrix_binary(predictions, test_labels))