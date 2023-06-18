from mlpr import *

if __name__ == "__main__":
    predicted_scores = np.load("Data/commedia_llr_infpar.npy")
    true_labels = np.load("Data/commedia_labels_infpar.npy")
    plot_roc_curve(predicted_scores, true_labels)