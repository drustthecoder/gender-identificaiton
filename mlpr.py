import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.datasets
from tqdm import tqdm
from time import sleep
import os


def load_data(file):
    matrix = []
    labels = []
    lines = open(file, "r")
    for l in lines:
        tokens = l.split(",")
        attrVector = column(np.array(tokens[0:-1], dtype=float))
        label = tokens[-1]

        labels.append(label)
        matrix.append(attrVector)
    D = np.hstack(matrix)
    L = np.array(labels, dtype=np.int32)
    return D, L

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


def sigmoid(z):
    # Sigmoid function
    return 1 / (1 + np.exp(-z))


def vrow(v):
    return v.reshape((1, v.size))


def vcol(v):
    return v.reshape(v.size, 1)

mrow = vrow

mcol = vcol


def histogram(samples, labels):
    class0Mask = labels == 0
    class1Mask = labels == 1


    samplesClass0 = samples[:, class0Mask]
    samplesClass1 = samples[:, class1Mask]
    for feature in range(samples.shape[0]):
        plt.hist(samplesClass0[feature,:], color="orange", density=True, bins=100, edgecolor='k', alpha = 0.4, label="Male")
        plt.hist(samplesClass1[feature,:], color="purple", density=True, bins=100, edgecolor='k', alpha = 0.4,label="Female")
        plt.xlabel("Feature %d" % feature)
        plt.legend()
        plt.title("Distribution of Feature %d" % feature)
        plt.savefig('plots/hist_feature_%d.png' % feature)
        plt.show()


def heatmap(D, title, color):
    plt.figure()
    pearson_matrix = np.corrcoef(D)
    plt.imshow(pearson_matrix, cmap=color, vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of %s Class Data" % title)
    plt.savefig("plots/heatmap_%s.png" % title)
    plt.show()


def scatter(samples, labels):
    class0Mask = labels == 0
    class1Mask = labels == 1

    samplesClass0 = samples[:, class0Mask]
    samplesClass1 = samples[:, class1Mask]
    for i in tqdm(range(samples.shape[0])):
        for j in range(samples.shape[0]):
            if i == j:
                continue
            else:
                plt.scatter(samplesClass0[i, :], samplesClass0[j, :], color="orange", label="Male", alpha=0.2,  s=80)
                plt.scatter(samplesClass1[i, :], samplesClass1[j, :], color="purple", label="Female", alpha=0.2,  s=80)
                plt.xlabel("feature_%d" % j)
                plt.ylabel("feature_%d" % i)
                plt.legend()
                plt.savefig('plots/scatter_feature_%d_%d.png' % (i, j))
                plt.show()


def apply_PCA(data, num_components):
    if num_components == "-":
        return data
    principal_components = PCA(data, num_components)
    return np.dot(principal_components.T, data)


def PCA(data, num_components):
    # Calculate the mean of each row in the data matrix
    mean_vector = np.mean(data, axis=1)

    # Center the data by subtracting the mean from each column
    centered_data = data - mean_vector.reshape(-1, 1)

    # Calculate the covariance matrix
    average_covariance_matrix = np.dot(centered_data, centered_data.T) / data.shape[1]

    # Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(average_covariance_matrix)

    # Sort the eigenvectors in descending order based on their corresponding eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'num_components' eigenvectors as the principal components
    principal_components = sorted_eigenvectors[:, 0:num_components]

    return principal_components


def plot_explained_variance_range(data, min_dim, max_dim):
    # Calculate the mean of each row in the data matrix
    mean_vector = np.mean(data, axis=1)

    # Center the data by subtracting the mean from each column
    centered_data = data - mean_vector.reshape(-1, 1)

    # Calculate the covariance matrix
    average_covariance_matrix = np.dot(centered_data, centered_data.T) / data.shape[1]

    # Perform eigendecomposition on the covariance matrix
    eigenvalues, _ = np.linalg.eigh(average_covariance_matrix)

    # Sort the eigenvalues in descending order
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

    # Initialize arrays to store the PCA dimensions and the corresponding explained variances
    dimensions = []
    explained_variances = []

    # Calculate the fraction of explained variance for each PCA dimension in the range
    for num_components in range(min_dim, max_dim + 1):
        total_variance = np.sum(sorted_eigenvalues)
        explained_variance = np.sum(sorted_eigenvalues[:num_components]) / total_variance

        dimensions.append(num_components)
        explained_variances.append(explained_variance)

    print(list(zip(dimensions, explained_variances)))
    # Plot the fraction of explained variance
    plt.plot(dimensions, explained_variances, 'bo-')
    plt.xlabel('PCA Dimension')
    plt.ylabel('Fraction of Explained Variance')
    plt.title('Fraction of Explained Variance by PCA Dimension')
    plt.xticks(list(range(0,13)))
    plt.yticks(list(i/10 for i in range(0, 11)))
    plt.grid()
    plt.savefig("plots/explained_variance.png")
    plt.show()

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
        # print("Optimized weights and bias:", optimized_weights_and_bias)
        # print("Optimal loss:", optimal_loss)

        # Compute predictions on the training data
        weights = optimized_weights_and_bias[:self.num_features]
        bias = optimized_weights_and_bias[-1]

        # compute scores using evaluation data
        scores = np.dot(weights.T, self.evaluation_data) + bias
        # predictions = np.sign(scores)
        #
        # # Compute the error rate
        # error_rate = calculate_error_rate(predictions, self.evaluation_labels)

        return scores


def confusion_matrix_binary(predictions, true_labels):
    assert len(predictions) == len(true_labels), "Input lists must have the same length."

    # matrix = np.zeros((2, 2), dtype=int)
    #
    # for pred, true_label in zip(predictions, true_labels):
    #     matrix[pred, true_label] += 1
    #
    # return matrix
    # Count the occurrences of each unique combination of true labels and predictions

    # optimized version

    # Count the occurrences of each unique combination of true labels and predictions
    counts = np.bincount(true_labels + 2 * predictions, minlength=4)

    # Reshape the count array into a 2x2 confusion matrix
    matrix = counts.reshape((2, 2))

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


def compute_min_DCF(scores, true_labels, cost_params):
    min_dcf = float('inf')
    # dcf_list = []
    sorted_scores = np.sort(scores)
    for threshold in sorted_scores:
        predictions = np.array([0 if llr <= threshold else 1 for llr in scores])
        conf_mat = confusion_matrix_binary(predictions, true_labels)
        dcf = compute_DCF_from_conf_mat(conf_mat, cost_params)
        min_dcf = dcf if dcf < min_dcf else min_dcf
    return min_dcf
    #     dcf_list.append(dcf)
    # return np.array(dcf_list).min()


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


def column(v):
    return v.reshape((v.size), 1)


def row(v):
    return v.reshape(1, v.size)


class SupportVectorMachines:
    def __init__(self, C, mode, pT, gamma=1, d=2, K=1):
        self.C = C
        self.mode = mode
        self.pT = pT
        self.d = d
        self.gamma = gamma
        self.K = K
        self.w_start = None
        self.H = None

    def train(self, DTR, LTR):
        DTRext = np.vstack([DTR, np.ones((1, DTR.shape[1]))])

        DTR0 = DTR[:, LTR == 0]
        DTR1 = DTR[:, LTR == 1]
        nF = DTR0.shape[1]
        nT = DTR1.shape[1]
        emp_prior_F = (nF / DTR.shape[1])
        emp_prior_T = (nT / DTR.shape[1])
        Cf = self.C * self.pT / emp_prior_F
        Ct = self.C * self.pT / emp_prior_T

        Z = np.zeros(LTR.shape)
        Z[LTR == 0] = -1
        Z[LTR == 1] = 1

        if self.mode == "linear":
            H = np.dot(DTRext.T, DTRext)
            H = column(Z) * row(Z) * H
        elif self.mode == "polynomial":
            H = np.dot(DTRext.T, DTRext) ** self.d
            H = column(Z) * row(Z) * H
        elif self.mode == "RBF":
            dist = column((DTR ** 2).sum(0)) + row((DTR ** 2).sum(0)) - 2 * np.dot(DTR.T, DTR)
            H = np.exp(-self.gamma * dist) + self.K
            H = column(Z) * row(Z) * H

        self.H = H

        bounds = [(-1, -1)] * DTR.shape[1]
        for i in range(DTR.shape[1]):
            if LTR[i] == 0:
                bounds[i] = (0, Cf)
            else:
                bounds[i] = (0, Ct)

        alpha_star, x, y = scipy.optimize.fmin_l_bfgs_b(
            self._LDual,
            np.zeros(DTR.shape[1]),
            # bounds = [(0, self.C)] * DTR.shape[1],
            bounds=bounds,
            factr=1e7,
            maxiter=100000,
            maxfun=100000
        )

        self.w_star = np.dot(DTRext, column(alpha_star) * column(Z))

    def compute_scores(self, DTE):
        DTEext = np.vstack([DTE, np.ones((1, DTE.shape[1]))])
        S = np.dot(self.w_star.T, DTEext)
        return S.ravel()

    def _JDual(self, alpha):
        Ha = np.dot(self.H, column(alpha))
        aHa = np.dot(row(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)

    def _LDual(self, alpha):
        loss, grad = self._JDual(alpha)
        return -loss, -grad

    def _JPrimal(self, DTRext, w, Z):
        S = np.dot(row(w), DTRext)
        loss = np.maximum(np.zeros(S.shape), 1 - Z * S).sum()
        return 0.5 * np.linalg.norm(w) ** 2 + self.C * loss


def get_fold_data(dataset, labels, k, fold_number, seed=42):
    num_samples = dataset.shape[1]
    fold_size = num_samples // k

    # Set the random seed
    np.random.seed(seed)

    # Shuffle the data and labels
    shuffled_indices = np.random.permutation(num_samples)
    shuffled_data = dataset[:, shuffled_indices]
    shuffled_labels = labels[shuffled_indices]

    start = (fold_number - 1) * fold_size
    end = start + fold_size

    test_data = shuffled_data[:, start:end]
    train_data = np.concatenate((shuffled_data[:, :start], shuffled_data[:, end:]), axis=1)

    test_labels = shuffled_labels[start:end]
    train_labels = np.concatenate((shuffled_labels[:start], shuffled_labels[end:]))

    return train_data, train_labels, test_data, test_labels


def MVG(DTE, DTR, LTR):
    h = {}

    for i in range(2):
        mu, C = estimate_mean_and_covariance(DTR[:, LTR == i])
        h[i] = (mu, C)

    SJoint = np.zeros((2, DTE.shape[1]))
    logSJoint = np.zeros((2, DTE.shape[1]))
    dens = np.zeros((2, DTE.shape[1]))
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu, C = h[label]
        dens[label, :] = np.exp(logpdf_GAU_ND_fast(DTE, mu, C).ravel())
        SJoint[label, :] = dens[label, :] * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND_fast(DTE, mu, C).ravel() + np.log(classPriors[label])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = np.exp(logPost)
    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2, np.log(dens[1] / dens[0])


def naive_MVG(DTE, DTR, LTR):
    h = {}

    for i in range(2):
        mu, C = estimate_mean_and_covariance(DTR[:, LTR == i])
        C = C * np.identity(C.shape[0])
        h[i] = (mu, C)

    SJoint = np.zeros((2, DTE.shape[1]))
    logSJoint = np.zeros((2, DTE.shape[1]))
    dens = np.zeros((2, DTE.shape[1]))
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu, C = h[label]
        dens[label, :] = np.exp(logpdf_GAU_ND_fast(DTE, mu, C).ravel())
        SJoint[label, :] = dens[label, :] * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND_fast(DTE, mu, C).ravel() + np.log(classPriors[label])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = np.exp(logPost)

    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2, np.log(dens[1] / dens[0])


def tied_cov_GC(DTE, DTR, LTR):
    h = {}
    Ctot = 0
    for i in range(2):
        mu, C = estimate_mean_and_covariance(DTR[:, LTR == i])
        Ctot += DTR[:, LTR == i].shape[1] * C
        h[i] = (mu)

    Ctot = Ctot / DTR.shape[1]

    SJoint = np.zeros((2, DTE.shape[1]))
    logSJoint = np.zeros((2, DTE.shape[1]))
    dens = np.zeros((2, DTE.shape[1]))
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu = h[label]
        dens[label, :] = np.exp(logpdf_GAU_ND_fast(DTE, mu, Ctot).ravel())
        SJoint[label, :] = dens[label, :] * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND_fast(DTE, mu, Ctot).ravel() + np.log(classPriors[label])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = np.exp(logPost)

    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2, np.log(dens[1] / dens[0])


def tied_cov_naive_GC(DTE, DTR, LTR):
    h = {}
    Ctot = 0
    for i in range(2):
        mu, C = estimate_mean_and_covariance(DTR[:, LTR == i])
        Ctot += DTR[:, LTR == i].shape[1] * C
        h[i] = (mu)

    Ctot = Ctot / DTR.shape[1]
    Ctot = Ctot * np.identity(Ctot.shape[0])

    SJoint = np.zeros((2, DTE.shape[1]))
    logSJoint = np.zeros((2, DTE.shape[1]))
    dens = np.zeros((2, DTE.shape[1]))
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu = h[label]
        dens[label, :] = np.exp(logpdf_GAU_ND_fast(DTE, mu, Ctot).ravel())
        SJoint[label, :] = dens[label, :] * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND_fast(DTE, mu, Ctot).ravel() + np.log(classPriors[label])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = np.exp(logPost)

    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2, np.log(dens[1] / dens[0])


def weighted_logreg_obj_wrap(DTR, LTR, l, pi=0.5):
    M = DTR.shape[0]
    Z = LTR * 2.0 - 1.0

    def logreg_obj(v):
        w = mcol(v[0:M])
        b = v[-1]
        reg = 0.5 * l * np.linalg.norm(w) ** 2
        s = (np.dot(w.T, DTR) + b).ravel()
        nt = DTR[:, LTR == 0].shape[1]
        avg_risk_0 = (np.logaddexp(0, -s[LTR == 0] * Z[LTR == 0])).sum()
        avg_risk_1 = (np.logaddexp(0, -s[LTR == 1] * Z[LTR == 1])).sum()
        return reg + (pi / nt) * avg_risk_1 + (1 - pi) / (DTR.shape[1] - nt) * avg_risk_0

    return logreg_obj


def weighted_logistic_reg_score(DTR, LTR, DTE, l, pi=0.5):
    logreg_obj = weighted_logreg_obj_wrap(np.array(DTR), LTR, l, pi)
    minimizer, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad=True)
    weights = minimizer[0:DTR.shape[0]]
    bias = minimizer[-1]
    scores = np.dot(weights.T, DTE) + bias
    return scores


def znorm(DTR, DTE):
    mu_DTR = mcol(DTR.mean(1))
    std_DTR = mcol(DTR.std(1))

    DTR_z = (DTR - mu_DTR) / std_DTR
    DTE_z = (DTE - mu_DTR) / std_DTR
    return DTR_z, DTE_z