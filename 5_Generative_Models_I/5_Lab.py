from mlpr import *

def classify_log_iris():
    # Two steps: 1. Training, 2. Classification
    data, labels = load_iris()
    (train_data, train_labels), (test_data, test_labels) = split_db_2to1(data, labels)

    # Training: Calculate the mean and covariance matrix of each class
    class_params = {}  # Dictionary to store parameters for each class
    for class_label in [0, 1, 2]:  # Iterate over each class label
        class_data = train_data[:, train_labels == class_label]  # Select data of this class
        class_params[class_label] = estimate_mean_and_covariance(class_data)

    # Classification
    prior_probs = np.log(vcol(np.ones(3) / 3.0))  # Log prior probability for each class (one third)
    # Each row is the class-conditional density for test samples given the hypothesized class
    class_conditional_densities = []
    for hypothesized_class in [0, 1, 2]:  # Iterate over each hypothesized class
        mean, covariance = class_params[hypothesized_class]  # Get the model parameters of the class

        # Compute the log-density of samples belonging to the class, resulting in a column vector
        class_conditional_density = logpdf_GAU_ND_fast(test_data, mean, covariance)

        class_conditional_densities.append(vrow(class_conditional_density))

    # Convert class_conditional_densities from an array to a matrix of class-conditional log densities
    # Each column of class_conditional_densities consists of the log densities of a sample in each class
    class_conditional_densities = np.vstack(class_conditional_densities)
    class_conditional_densities = class_conditional_densities + prior_probs  # Add log prior to class-conditional log densities

    # logsumexp(class_conditional_densities, 0) performs exponentiation of class_conditional_densities (log-joint),
    # then sums the joint matrix over the rows, and finally takes the logarithm.
    # logsumexp is numerically stable compared to log(exp(s).sum(0))
    # log_posterior_probabilities is the log-joint probability minus the log-marginal probability
    log_posterior_probabilities = class_conditional_densities - vrow(scipy.special.logsumexp(class_conditional_densities, 0))
    posterior_probabilities = np.exp(log_posterior_probabilities)

    predicted_labels = np.argmax(posterior_probabilities, axis=0)
    true_labels = test_labels
    correct_predictions = predicted_labels == true_labels
    num_correct_predictions = np.sum(correct_predictions)
    total_predictions = true_labels.size
    logging.info(f"Accuracy: {num_correct_predictions / total_predictions}, Correct Predictions: {num_correct_predictions}, Total Predictions: {total_predictions}")
    return class_conditional_densities, posterior_probabilities


logging.basicConfig(level=logging.INFO)
# S, P = classify_iris()
classify_log_iris()
# print(f"S: {S}")
# print(f"P: {P}")

