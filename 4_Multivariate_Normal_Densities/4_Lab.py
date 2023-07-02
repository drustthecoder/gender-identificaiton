from mlpr import *
import matplotlib.pyplot as plt

# D, L = load_iris()
# mu, C = mean_cov_estimate(D)
# sample = vcol(D[:, 0])
# print(logpdf_GAU_ND_1Sample(sample, mu, C))


XPlot = np.linspace(-8, 12, 1000)
m = np.ones((1, 1)) * 1.0
C = np.ones((1, 1)) * 2.0
plt.figure()
plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND_fast(vrow(XPlot), m, C)))
plt.title('plot1')
plt.show()

XND = np.load('Solution/XND.npy')
mu = np.load('Solution/muND.npy')
C = np.load('Solution/CND.npy')
pdfSol = np.load('Solution/llND.npy')
pdfGau = logpdf_GAU_ND_fast(XND, mu, C)
print(f"bigger absolute of pdfSol vs. pdfGau: {np.abs(pdfSol - pdfGau).max()}")

# ML estimate mean and covariance for XND
m_ML, C_ML = estimate_mean_and_covariance(XND)
llXND = loglikelihood(XND, m_ML, C_ML)
print(f"XND loglikelihood: {llXND}")

# ML estimate mean and covariance for X1D
X1D = np.load('Solution/X1D.npy')
m_ML, C_ML = estimate_mean_and_covariance(X1D)
XPlot = np.linspace(-8, 12, 1000)

# we wanted to find mean and covariance matrix that
# maximize the log-likelihood function
# computing the log-likelihood for other values of µ and Σ would results in a lower
# value of the log-likelihood
llX1D = loglikelihood(X1D, m_ML, C_ML)
print(f"X1D loglikelihood: {llX1D}")

plt.figure()
plt.hist(X1D.ravel(), bins=50, density=True)
plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND_fast(vrow(XPlot), m_ML, C_ML)))
plt.title('plot2')
plt.show()

