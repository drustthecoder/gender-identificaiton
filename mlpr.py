import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.datasets
from tqdm import tqdm


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


def vrow(v):
    return v.reshape((1, v.size))


def vcol(v):
    return v.reshape(v.size, 1)


def PCA(data, num_components):
    # Calculate the mean of each row in the data matrix
    mean_vector = np.mean(data, axis=1)

    # Center the data by subtracting the mean from each column
    centered_data = data - mean_vector.reshape(-1, 1)

    # Calculate the covariance matrix
    covariance_matrix = np.dot(centered_data, centered_data.T) / data.shape[1]

    # Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort the eigenvectors in descending order based on their corresponding eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'num_components' eigenvectors as the principal components
    principal_components = sorted_eigenvectors[:, 0:num_components]

    return principal_components


def calculate_scatter_matrices(dataset, labels):
    """
    Calculates the scatter matrices SB (between-class scatter matrix) and
    SW (within-class scatter matrix) for a given dataset and its corresponding labels.
    """
    # Initialize SB and SW scatter matrices
    SB = 0
    SW = 0

    # Calculate the mean vector of the dataset
    mean_vector = np.expand_dims(dataset.mean(axis=1), axis=1)

    # Iterate over each class
    for i in range(labels.max() + 1):
        # Select data points belonging to the current class
        data_class = dataset[:, labels == i]

        # Calculate the mean vector of the current class
        class_mean_vector = vcol(data_class.mean(1))

        # Update the SW scatter matrix by summing the outer product of the centered data points
        SW += np.dot(data_class - class_mean_vector, (data_class - class_mean_vector).T)

        # Update the SB scatter matrix by summing the outer product of the centered class mean vector
        SB += data_class.shape[1] * np.dot(class_mean_vector - mean_vector, (class_mean_vector - mean_vector).T)

    # Normalize the SW and SB scatter matrices by dividing by the number of data points
    SW /= dataset.shape[1]
    SB /= dataset.shape[1]

    # Return the calculated SB and SW scatter matrices
    return SB, SW


def LDA(data, labels, num_components):
    # Calculate scatter matrices
    scatter_between, scatter_within = calculate_scatter_matrices(data, labels)

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = scipy.linalg.eigh(scatter_between, scatter_within)

    # Select the desired number of components
    selected_eigenvectors = eigenvectors[:, ::-1][:, 0:num_components]

    # Return the eigenvectors corresponding to the desired number of components
    return selected_eigenvectors


def logpdf_GAU_ND_fast(X, mean, covariance):
    # calculates the logarithm of the probability density function (PDF)
    # for a multivariate Gaussian distribution.

    # Calculate the difference between X and the mean
    X_centered = X - mean

    # Get the number of dimensions
    M = X.shape[0]

    # Calculate the constant term in the logarithm
    const = -0.5 * M * np.log(2 * np.pi)

    # Calculate the logarithm of the determinant of the covariance matrix
    logdet = np.linalg.slogdet(covariance)[1]

    # Calculate the inverse of the covariance matrix
    inverse_covariance = np.linalg.inv(covariance)

    # Calculate the vector v by multiplying X_centered with the inverse_covariance
    v = (X_centered * np.dot(inverse_covariance, X_centered)).sum(0)

    # Calculate the logarithm of the PDF
    log_pdf = const - 0.5 * logdet - 0.5 * v

    return log_pdf


def estimate_mean_and_covariance(data):
    mean = vcol(data.mean(1))
    cov = np.dot(data - mean, (data - mean).T) / data.shape[1]
    return mean, cov


def loglikelihood(data, mean, covariance):
    return np.sum(logpdf_GAU_ND_fast(data, mean, covariance))


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def calculate_error_rate(predictions, labels):
    incorrect_predictions = np.sum(predictions != labels)
    error_rate = incorrect_predictions / len(labels)
    return error_rate


class LogisticRegressionModel:
    def __init__(self, training_data, training_labels, evaluation_data, evaluation_labels, regularization_param):
        self.training_data = training_data
        self.training_labels = training_labels * 2.0 - 1.0  # This is ZTR
        self.evaluation_data = evaluation_data
        self.evaluation_labels = evaluation_labels * 2.0 - 1.0
        self.regularization_param = regularization_param  # This is Lambda
        self.num_features = training_data.shape[0]  # This is dim

    def calculate_loss(self, weights_and_bias):
        weights = np.array(weights_and_bias[:self.num_features])
        bias = weights_and_bias[-1]
        scores = np.dot(weights.T, self.training_data) + bias
        loss_per_sample = np.logaddexp(0, -self.training_labels * scores)
        loss = loss_per_sample.mean() + 0.5 * self.regularization_param * np.linalg.norm(weights) ** 2
        return loss

    def train(self):
        initial_weights_and_bias = np.zeros(self.num_features + 1)
        optimized_weights_and_bias, optimal_loss, optimization_info = scipy.optimize.fmin_l_bfgs_b(
            self.calculate_loss,
            x0=initial_weights_and_bias,
            approx_grad=True
        )
        print("Optimized weights and bias:", optimized_weights_and_bias)
        print("Optimal loss:", optimal_loss)

        # Compute predictions on the training data
        weights = optimized_weights_and_bias[:self.num_features]
        bias = optimized_weights_and_bias[-1]

        # compute scores using evaluation data
        scores = np.dot(weights.T, self.evaluation_data) + bias
        predictions = np.sign(scores)

        # Compute the error rate
        error_rate = calculate_error_rate(predictions, self.evaluation_labels)

        return optimized_weights_and_bias, optimal_loss, error_rate


def confusion_matrix_binary(predictions, true_labels):
    assert len(predictions) == len(true_labels), "Input lists must have the same length."

    matrix = np.zeros((2, 2), dtype=int)

    for pred, true_label in zip(predictions, true_labels):
        matrix[pred, true_label] += 1

    return matrix


def compute_optimal_bayes_decisions(log_likelihoods, cost_params):
    # Computes optimal Bayes decisions based on
    # log-likelihoods and cost matrix parameters tuple (π1, Cfn, Cfp)

    # Unpacking cost matrix parameters
    pi_1, Cfn, Cfp = cost_params

    # Compute the threshold
    threshold = -np.log((pi_1 * Cfn) / ((1 - pi_1) * Cfp))

    # Compute prediction by doing optimal Bayes decision
    # Choose the class c∗ which has minimum expected Bayes cost
    predictions = np.array([0 if llr <= threshold else 1 for llr in log_likelihoods])
    return predictions


def compute_fnr_fpr(confusion_mat):
    fnr = confusion_mat[0, 1] / (confusion_mat[0, 1] + confusion_mat[1, 1])
    fpr = confusion_mat[1, 0] / (confusion_mat[0, 0] + confusion_mat[1, 0])
    return fnr, fpr


def compute_DCF_from_conf_mat(conf_mat, cost_params):
    # Compute empirical Bayes risk (or detection cost function, DCF)
    pi_1, Cfn, Cfp = cost_params
    fnr, fpr = compute_fnr_fpr(conf_mat)
    dcf = pi_1 * Cfn * fnr + (1 - pi_1) * Cfp * fpr
    dummy_risk = min([pi_1 * Cfn, (1 - pi_1) * Cfp])
    normalized_dcf = dcf / dummy_risk
    return normalized_dcf


def compute_min_DCF(log_likelihoods, true_labels, cost_params):
    min_dcf = float('inf')
    for threshold in sorted(log_likelihoods):
        predictions = np.array([0 if llr <= threshold else 1 for llr in log_likelihoods])
        conf_mat = confusion_matrix_binary(predictions, true_labels)
        dcf = compute_DCF_from_conf_mat(conf_mat, cost_params)
        min_dcf = dcf if dcf < min_dcf else min_dcf
    return min_dcf


def plot_roc_curve(log_likelihoods, true_labels):
    tpr_values, fpr_values = [], []
    for threshold in sorted(log_likelihoods):
        predictions = np.array([0 if llr <= threshold else 1 for llr in log_likelihoods])
        conf_mat = confusion_matrix_binary(predictions, true_labels)
        fnr, fpr = compute_fnr_fpr(conf_mat)
        tpr = 1 - fnr
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    plt.plot(fpr_values, tpr_values)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.show()


def plot_bayes_error(models, effPriorLogOdds):
    # models is an array and each element contains name, log_likelihoods, true_labels
    for m in tqdm(models, desc="plot bayes error..."):
        name = m[0]
        log_likelihoods = m[1]
        true_labels = m[2]
        dcf = []
        mindcf = []
        # effPriorLogOdds = np.linspace(-3, 3, 21)
        for p in effPriorLogOdds:
            pi_tilde = 1 / (1 + np.exp(-p))
            app = (pi_tilde, 1, 1)
            optimal_bayes_predictions = compute_optimal_bayes_decisions(log_likelihoods, app)
            optimal_bayes_conf_mat = confusion_matrix_binary(optimal_bayes_predictions, true_labels)
            dcf.append(compute_DCF_from_conf_mat(optimal_bayes_conf_mat, app))
            mindcf.append(compute_min_DCF(log_likelihoods, true_labels, app))
        plt.plot(effPriorLogOdds, dcf, label=f"DCF {name}")
        plt.plot(effPriorLogOdds, mindcf, label=f"min DCF {name}")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF value')
    plt.title('Bayes error plot')
    plt.legend(loc='lower left')
    plt.show()


def logpdf_gmm(x, gmmList):
    # Compute the joint log-density of each data point
    joint_log_densities = []
    for weight, mean, covariance in gmmList:
        # Compute the cluster-conditional densities + logarithm of prior
        log_dens = logpdf_GAU_ND_fast(x, mean, covariance) + np.log(weight)
        joint_log_densities.append(vrow(log_dens))
    y = np.vstack(joint_log_densities)

    # Compute the log-marginal probability of each data point
    return scipy.special.logsumexp(joint_log_densities, axis=0)


def lbg_double_cluster_count_gmm(gmm):
    updated_gmm = []
    for (weight, mean, covariance) in gmm:
        U, s, _ = np.linalg.svd(covariance)

        # Create two new Gaussian components by perturbing the mean along the first principal component
        updated_gmm.append((weight * 0.5, mean + s[0] ** 0.5 * U[:, 0:1] * 0.1, covariance))
        updated_gmm.append((weight * 0.5, mean - s[0] ** 0.5 * U[:, 0:1] * 0.1, covariance))

    return updated_gmm


def expectation_maximization_gmm(data, gmm, diagCov=False, nEMIters=10, tiedCov=False):
    log_likelihood = None
    previous_log_likelihood = None
    delta_LL = 1.0
    iteration = 0
    starting_log_likelihood = None

    while delta_LL > 1e-6:
        joint_log_densities = []
        for weight, mean, covariance in gmm:
            log_likelihood_comp = logpdf_GAU_ND_fast(data, mean, covariance) + np.log(weight)
            joint_log_densities.append(vrow(log_likelihood_comp))
        joint_log_densities_array = np.vstack(joint_log_densities)

        log_marginal = scipy.special.logsumexp(joint_log_densities_array, axis=0)

        # Compute the posterior probabilities (gammas) of each sample in each cluster
        # By dividing joing log density by log marginal (subtract in log domain)
        log_posterior = joint_log_densities_array - log_marginal
        posterior = np.exp(log_posterior)

        previous_log_likelihood = log_likelihood

        # Compute the log-likelihood as the sum over the log-marginals
        log_likelihood = log_marginal.sum()
        if starting_log_likelihood is None:
            starting_log_likelihood = log_likelihood

        # Compute the delta between the previous and current log-likelihood
        if previous_log_likelihood is not None:
            delta_LL = log_likelihood - previous_log_likelihood
        iteration += 1

        updated_gmm = []
        for i in range(posterior.shape[0]):
            # Compute zeroth-order statistics: sum of gammas for cluster i
            Z = posterior[i].sum()

            # Compute first-order statistics: delta multiplied by weight given by posterior values summed over columns
            F = vcol((posterior[i:i + 1, :] * data).sum(1))
            S = np.dot((posterior[i:i + 1, :] * data), data.T)

            # Update the weight, mean, and covariance
            weight_update = Z / data.shape[1]
            mean_update = F / Z
            covariance_update = S / Z - np.dot(mean_update, mean_update.T) + np.eye(covariance.shape[0]) * 1e-9

            # If covariance needs to be diagonal, get the diagonal of covariance
            if diagCov:
                covariance_update = covariance_update * np.eye(covariance_update.shape[0])

            updated_gmm.append((weight_update, mean_update, covariance_update))

        if tiedCov:
            total_covariance = sum([weight * covariance for weight, mean, covariance in updated_gmm])
            updated_gmm = [(weight, mean, total_covariance) for weight, mean, covariance in updated_gmm]

        gmm = updated_gmm
    print(f"iters: {iteration}, Log-likelihood start: {starting_log_likelihood}, end: {log_likelihood}, gmm len: {len(gmm)}")
    return gmm


if __name__ == "__main__":
    print("This is mlpr.py")
