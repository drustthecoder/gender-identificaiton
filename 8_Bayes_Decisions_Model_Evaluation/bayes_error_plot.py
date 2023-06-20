from mlpr import *

if __name__ == "__main__":
    predicted_scores = np.load("Data/commedia_llr_infpar.npy")
    predicted_scores_eps1 = np.load("Data/commedia_llr_infpar_eps1.npy")
    true_labels = np.load("Data/commedia_labels_infpar.npy")
    # plot_bayes_error(predicted_scores, true_labels)
    # plot_bayes_error(predicted_scores_eps1, true_labels)
    effPriorLogOdds = np.linspace(-3, 3, 21)
    plot_bayes_error(
        [
            ["(E = 0.001)", predicted_scores, true_labels],
            ["(E = 1)", predicted_scores_eps1, true_labels]
        ],
        effPriorLogOdds
    )
    # print(len(predicted_scores))
    # print(len(list(set(predicted_scores))))