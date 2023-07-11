from mlpr import *

# PCA 6 keeps more than 90 percent of dataset variability

data, labels = load_data("data/Test.txt")

#preprocess?
# center data
# Standardize variances
# Whiten the covariance matrix
# L2 (or length) normalization

pca_values = ["-", 12, 11, 10, 9]
results = []
app = (0.5, 1, 1)
k = 5
gaussian_funcs = {
    "MVG": MVG,
    "naive_MVG": naive_MVG,
    "tied_cov_GC": tied_cov_GC
}

for func in ["MVG", "naive_MVG", "tied_cov_GC"]:
# for func in ["naive_MVG"]:
    print(f"{func} classifier")
    print(f"PCA\t\tminDCF(pi=0.5)")
    for pca_num in pca_values:
        for num_fold in list(range(1, 6)):
            transformed_data = apply_PCA(data, pca_num)
            train_data, train_labels, test_data, test_labels = get_fold_data(transformed_data, labels, k, num_fold)
            _, _, scores = gaussian_funcs[func](test_data, train_data, train_labels)
            mindcf = compute_min_DCF(scores, test_labels, app)
            results.append(mindcf)
        print(f"{pca_num}\t\t{round(np.array(results).mean(), 3)}")
        results = []


# for num in pca_values:
#     transformed_data = apply_PCA(data, num)
# print(labels[0:3000].sum())
# print(labels[3000:].sum())
# for i in [1,2,3]:
#     train_data, train_labels, test_data, test_labels = get_fold_data(data, labels, 3, 1)
#     print(test_labels.sum())
# train_data, train_labels, test_data, test_labels = get_fold_data(data, labels, 3, 1)

# print(scores.sum())
# print(labels.sum())
# print((((scores>0)==test_labels)*1.0).sum())


