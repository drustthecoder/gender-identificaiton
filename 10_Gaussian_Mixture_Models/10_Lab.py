import logging

from mlpr import *
from Data.GMM_load import load_gmm
from pprint import pprint


samples_4_d = np.load("Data/GMM_data_4D.npy")
gmm_refrence = load_gmm("Data/GMM_4D_3G_init.json")
log_densities_all_samples = np.load("Data/GMM_4D_3G_init_ll.npy")
GMM_4D_3G_EM = load_gmm("Data/GMM_4D_3G_EM.json")

# Check your log-density function provides the same results
log_densities_calculated = logpdf_GMM(samples_4_d, gmm_refrence)
print(f"log-density correctness out of 1000: {(log_densities_calculated == log_densities_all_samples).sum()}")
logging.basicConfig(level=logging.DEBUG)
gmm_calculated = ML_GMM_IT(samples_4_d, gmm_refrence, diagCov=False, nEMIters=10, tiedCov=False)
print(f"calculated gmm is:")
pprint(gmm_calculated)
