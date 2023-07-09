from mlpr import *
import xvalidator
import svm


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
    print(f"(pi_1, Cfn, Cfp)\t\tDCF\t\tmin DCF")
    for app in applications:
        predicted_labels = compute_optimal_bayes_decisions(commedia_llr_infpar, app)
        commedia_confusion_mat = confusion_matrix_binary(predicted_labels, commedia_labels_infpar)
        commedia_DCF = compute_DCF_from_conf_mat(commedia_confusion_mat, app)
        minimum_DCF = compute_min_DCF(commedia_llr_infpar, commedia_labels_infpar, app)
        min_DCF_xvalidator = xvalidator.compute_min_DCF(commedia_llr_infpar, commedia_labels_infpar, *app)
        print(f"{app}\t\t{commedia_DCF}\t\t{minimum_DCF}\t\t{min_DCF_xvalidator}")