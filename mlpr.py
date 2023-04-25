import numpy
import scipy


def vrow(v):
    return v.reshape((1, v.size))


def vcol(v):
    return v.reshape(v.size, 1)


def PCA(D, m):
    mu = vcol(D.mean(1))
    C = numpy.dot(D - mu, (D - mu).T) / D.shape[1]
    s, U = numpy.linalg.eigh(C)
    U = U[:, ::-1]
    P = U[:, 0:m]

    return P


def PCA2(D, m):
    mu = vcol(D.mean(1))
    C = numpy.dot(D - mu, (D - mu).T) / D.shape[1]
    U, _, _ = numpy.linalg.svd(C)
    P = U[:, 0:m]

    return P


def SbSw(D, L):
    SB = 0
    SW = 0
    mu = vcol(D.mean(1))
    for i in range(L.max() + 1):
        DCls = D[:, L == i]
        muCls = vcol(DCls.mean(1))
        SW += numpy.dot(DCls - muCls, (DCls - muCls).T)
        SB += DCls.shape[1] * numpy.dot(muCls - mu, (muCls - mu).T)
    SW /= D.shape[1]
    SB /= D.shape[1]
    return SB, SW


def LDA1(D, L, m):
    SB, SW = SbSw(D, L)
    s, U = scipy.linalg.eigh(SB, SW)
    return U[:, ::-1][:, 0:m]


def LDA2(D, L, m):

    SB, SW = SbSw(D, L)
    U, s, _ = numpy.linalg.svd(SW)
    P1 = numpy.dot(U, vcol(1.0/s**0.5)*U.T)
    SBTilde = numpy.dot(P1, numpy.dot(SB, P1.T))
    U, _, _ = numpy.linalg.svd(SBTilde)
    P2 = U[:, 0:m]
    return numpy.dot(P1.T, P2)


def logpdf_GAU_ND_1Sample(x, mu, C):
    xc = x - mu  # Centered x
    M = x.shape[0]  # The size of the feature vector
    const = - 0.5 * M * numpy.log(2 * numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]  # Index 1 is the value of the log determinant
    L = numpy.linalg.inv(C)  # Inverse of the covariance = precision matrix
    v = numpy.dot(xc.T, numpy.dot(L, xc)).ravel()  # Ravel so instead of 1x1 matrix it's a 1-d array with one element
    return const - 0.5 * logdet - 0.5 * v



def logpdf_GAU_ND(X, mu, C):
    Y = []
    for i in range(X.shape[1]):
        Y.append(logpdf_GAU_ND_1Sample(X[:, i:i + 1], mu, C))
    return numpy.array(Y).ravel()


def logpdf_GAU_ND_fast(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = - 0.5 * M * numpy.log(2 * numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]
    L = numpy.linalg.inv(C)
    v = (XC * numpy.dot(L, XC)).sum(0)
    # The dot product produces Lx1, Lx2, ...
    return const - 0.5 * logdet - 0.5 * v


def get_ml_mu_sigma(D):
    mu = vcol(D.mean(1))
    C = numpy.dot(D - mu, (D - mu).T) / D.shape[1]
    return mu, C


def loglikelihood(X, mu, C):
    return numpy.sum(logpdf_GAU_ND_fast(X, mu, C))