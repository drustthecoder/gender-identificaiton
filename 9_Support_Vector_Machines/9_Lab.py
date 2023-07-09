from mlpr import *

if __name__=="__main__":
    data, labels = load_iris_binary()
    (train_data, train_labels), (test_data, test_labels) = split_db_2to1(data, labels)

    print("SVM")
    # for pi in [0.5, 0.1, 0.9]:
    pi = 0.5
    s = SupportVectorMachines(1, "linear", 0.5)
    s.train(train_data, train_labels)
    scores = s.compute_scores(test_data)
    print(scores)
    # min_DCF = compute_min_DCF(scores, test_labels, (pi, 1, 1))
    # min_DCF_xvalidator = xvalidator.compute_min_DCF(scores, test_labels, pi, 1, 1)
    # print(f"mlpr:{min_DCF}, xvalidator:{min_DCF_xvalidator}")
    # print("")