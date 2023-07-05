import logging

from mlpr import *
from Data.GMM_load import load_gmm
from pprint import pprint

samples_4_d = np.load("Data/GMM_data_4D.npy")
gmm_refrence = load_gmm("Data/GMM_4D_3G_init.json")
log_densities_all_samples = np.load("Data/GMM_4D_3G_init_ll.npy")
GMM_4D_3G_EM = load_gmm("Data/GMM_4D_3G_EM.json")

# Check your log-density function provides the same results
log_densities_calculated = logpdf_gmm(samples_4_d, gmm_refrence)
print(f"log-density correctness out of 1000: {(log_densities_calculated == log_densities_all_samples).sum()}")

# logging.basicConfig(level=logging.DEBUG)
#
# # print("calculate gmm with gmm_refrence")
# # gmm_calculated = ML_GMM_IT(samples_4_d, gmm_refrence, diagCov=False, nEMIters=10, tiedCov=False)
#
# # print("calculate gmm with empirical mean and cov")
# mean, cov = estimate_mean_and_covariance(samples_4_d)
# gmm_calculated = [(1.0, mean, cov)]
# i = 1
# # calculate gmm with max 512 clusters
# while i < 512:
#     i *= 2
#     gmm_2x = lbg_double_cluster_count_gmm(gmm_calculated)
#     gmm_calculated = expectation_maximization_gmm(samples_4_d, gmm_2x, diagCov=False, nEMIters=10, tiedCov=False)
