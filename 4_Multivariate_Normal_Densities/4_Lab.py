from mlpr import *
import matplotlib.pyplot as plt

# D, L = load_iris()
# mu, C = mean_cov_estimate(D)
# sample = vcol(D[:, 0])
# print(logpdf_GAU_ND_1Sample(sample, mu, C))

plt.figure()
XPlot = np.linspace(-8, 12, 1000)
m = np.ones((1,1)) * 1.0
C = np.ones((1,1)) * 2.0
plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
plt.show()

XND = np.load('Solution/XND.npy')
mu = np.load('Solution/muND.npy')
C = np.load('Solution/CND.npy')
pdfSol = np.load('Solution/llND.npy')
pdfGau = logpdf_GAU_ND(XND, mu, C)
print(np.abs(pdfSol - pdfGau).max())

#ML
m_ML, C_ML = estimate_mean_and_covariance(XND)
ll = loglikelihood(XND, m_ML, C_ML)
print(ll)


X1D = np.load('Solution/X1D.npy')
m_ML, C_ML = estimate_mean_and_covariance(X1D)
plt.figure()
plt.hist(X1D.ravel(), bins=50, density=True)
XPlot = np.linspace(-8, 12, 1000)
plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m_ML, C_ML)))
plt.show()
ll = loglikelihood(X1D, m_ML, C_ML)
print(ll)