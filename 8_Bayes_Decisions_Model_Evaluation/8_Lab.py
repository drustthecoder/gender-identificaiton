from mlpr import *





if __name__ == "__main__":
    # predictions = np.load("Data/commedia_ll.npy ")
    # true_labels = np.load("Data/commedia_labels.npy")
    # predictions = np.argmax(predictions, axis=0)
    # true_labels = true_labels.ravel()
    # confusion_mat = confusion_matrix(true_labels, predictions)
    # print(confusion_mat)

    commedia_llr_infpar = np.load("Data/commedia_llr_infpar.npy")
    commedia_labels_infpar = np.load("Data/commedia_labels_infpar.npy")
    applications = [
        (0.5, 1, 1),
        (0.8, 1, 1),
        (0.5, 10, 1),
        (0.8, 1, 10)
        ]
    print(f"(pi_1, Cfn, Cfp)\t\tDCFu\t\tDCF")
    for app in applications:
        predicted_labels = compute_optimal_bayes_decisions(commedia_llr_infpar, app)
        commedia_confusion_mat = confusion_matrix_binary(predicted_labels, commedia_labels_infpar)
        commedia_bayes_risk = compute_DCF(commedia_confusion_mat, app)
        pi_1, Cfn, Cfp = app
        dummy_risk = np.array([pi_1 * Cfn, (1-pi_1) * Cfp]).min()
        normalized_risk = commedia_bayes_risk/dummy_risk
        print(f"{app}\t\t{commedia_bayes_risk}\t\t{normalized_risk}")

