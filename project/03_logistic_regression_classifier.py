import random

from mlpr import *
import datetime


pi_t_str = '\u03C0\u209C'  # U+209C SUBSCRIPT SMALL LETTER T
pi_tilde_str = '\u03C0\u0303'  # Combining U+0303 TILDE after U+03C0 PI

data, labels = load_data("data/Test.txt")
precision = 3
pca_values = ["-", 11, 10, 9]
results = []
mean_results = []
app = (0.5, 1, 1)
k = 5
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
pi_t_list = [0.1, 0.5, 0.9]


plt.title("Comparison between Log-Reg and Log-Reg (z-norm)")
z_data = z_norm(data)
pi_tilde = 0.5
for d, l in [[data, "Log-Reg"], [z_data, "Log-Reg (z-norm)"]]:
    print(l)
    print(f"Lambda\t\tminDCF({pi_tilde_str}=0.5)")
    for lam in lambdas:
        for num_fold in list(range(1, 6)):
            train_data, train_labels, test_data, test_labels = get_fold_data(d, labels, k, num_fold)
            scores = weighted_logistic_reg_score(train_data, train_labels, test_data, lam, pi_tilde)
            mindcf = compute_min_DCF(scores, test_labels, app)
            results.append(mindcf)
        mean = round(np.array(results).mean(), precision)
        mean_results.append(mean)
        print(f"{lam}\t\t{mean}")
        results = []

    plt.plot(lambdas, mean_results, label=l)
    mean_results = []
plt.xlabel('Lambda')
plt.ylabel('minDCF(\u03C0\u0303=0.5)')
plt.xscale("log")
plt.legend()
timestamp = int(datetime.datetime.now().timestamp())
plt.savefig('plots/logreg_lam_mindcf_%d.png' % timestamp)
plt.show()


# results:
# no z-norm
# regularization 1e-4

results = []
lam = 1e-4
print("Trying different values for pi_t")
print("(no z-norm) Log-Reg with Lambda = 1e-4")
print(f"{pi_t_str}\t\tminDCF({pi_tilde_str}=0.5)")
for pi_t in pi_t_list:
    for num_fold in list(range(1, 6)):
        train_data, train_labels, test_data, test_labels = get_fold_data(data, labels, k, num_fold)
        scores = weighted_logistic_reg_score(train_data, train_labels, test_data, lam, pi_t)
        mindcf = compute_min_DCF(scores, test_labels, app)
        results.append(mindcf)
    mean = round(np.array(results).mean(), precision)
    mean_results.append(mean)
    print(f"{pi_t}\t\t{mean}")
    results = []

results = []
lam = 1e-4
pi_t = 0.5
print("Trying different values for PCA")
print("(no z-norm) Log-Reg with Lambda = 1e-4")
print(f"PCA\t\tminDCF({pi_tilde_str}=0.5)")
for pca in pca_values:
    transformed_data = apply_PCA(data, pca)
    for num_fold in list(range(1, 6)):
        train_data, train_labels, test_data, test_labels = get_fold_data(transformed_data, labels, k, num_fold)
        scores = weighted_logistic_reg_score(train_data, train_labels, test_data, lam, pi_t)
        mindcf = compute_min_DCF(scores, test_labels, app)
        results.append(mindcf)
    mean = round(np.array(results).mean(), precision)
    mean_results.append(mean)
    print(f"{pca}\t\t{mean}")
    results = []