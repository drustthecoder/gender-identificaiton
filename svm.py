import numpy as np
from tqdm import tqdm
import scipy


def toCol(theArray:np.array):
    try:
        return theArray.reshape(theArray.size, 1)
    except Exception as e:
        print(e)


def toRow(theArray:np.array):
    try:
        return theArray.reshape(1,theArray.size)
    except Exception as e:
        print(e)


class SupportVectorMachine:
    def __init__(
            self,
            kfold=None,
            targetPrior=None,
            rbf=False,
            polynomial=False,
            C=1.0,
            gamma=1.0,
            polynomial_c=0.0,
            d=2,
            pca=False,
            zNorm=False,
            pcaM=10,
            tqdm=True,
            eval=False
    ):
        self.allFolds = kfold
        self.rbf = rbf
        self.polynomial = polynomial
        self.C = C
        self.polynomial_c = polynomial_c
        self.d = d
        self.tqdm = not tqdm
        self.targetPrior = targetPrior
        self.gamma = gamma
        self.scores = list()
        self.groundTrouthLabels = list()
        self.eval= eval
        self.pca = pca
        self.zNorm = zNorm
        self.pcaM = pcaM

        self.train()

    def train(self):
        for (Xtrain, Ytrain), (Xtest, Ytest) in tqdm(self.allFolds, disable=self.tqdm):
            # trainData, testData = preprocessing(
            #     Xtrain=Xtrain,
            #     Xtest=Xtest,
            #     zNorm=self.zNorm,
            #     pca=self.pca,
            #     pcaM=self.pcaM
            # )
            trainData, testData = Xtrain, Xtest
            self.testSet = testData
            self.testSetLabel = Ytest

            self.testSetExtended = np.vstack([self.testSet, np.ones((1, self.testSet.shape[1]))])

            self.trainSetExtended = np.vstack([trainData, np.ones((1, Xtrain.shape[1]))])
            self.trainSetLabel = Ytrain

            Z = np.zeros(self.trainSetLabel.shape)
            Z[self.trainSetLabel == 1] = 1
            Z[self.trainSetLabel == 0] = -1

            if not self.rbf and not self.polynomial:
                # print("linear")
                H = np.dot(self.trainSetExtended.T, self.trainSetExtended)
            else:
                if self.rbf:
                    H = self.rbf_kernel()
                else:
                    # print("poly")
                    H = self.polynominal_kernel()

            H = toCol(Z) * toRow(Z) * H

            def JDual(alpha):
                Ha = np.dot(H, toCol(alpha))
                aHa = np.dot(toRow(alpha), Ha)
                a1 = alpha.sum()
                return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)

            def lDual(alpha):
                loss, gradiant = JDual(alpha)
                return -loss, -gradiant

            def JPrimal(w):
                S = np.dot(toRow(w), self.trainSetExtended)
                loss = np.maximum(np.zeros(S.shape), 1 - Z * S).sum()
                return 0.5 * np.linalg.norm(w) ** 2 + self.C * loss

            if self.targetPrior is None:
                alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                    lDual,
                    np.zeros((self.trainSetExtended.shape[1])),
                    bounds=[(0, self.C)] * self.trainSetExtended.shape[1],
                    factr=1.0,
                    maxiter=100000,
                    maxfun=100000)
            else:
                mask = self.trainSetLabel == 1
                c1 = self.C * self.trainSetExtended.shape[1] * (self.targetPrior/self.trainSetExtended[:, self.trainSetLabel == 1].shape[1])
                c0 = self.C * self.trainSetExtended.shape[1] * ((1-self.targetPrior)/self.trainSetExtended[:, self.trainSetLabel == 0].shape[1])

                alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                    lDual,
                    np.zeros((self.trainSetExtended.shape[1])),
                    bounds= [(0, c1) if m else (0, c0) for m in mask],
                    factr=1.0,
                    maxiter=100000,
                    maxfun=100000)

            wStar = np.dot(self.trainSetExtended, toCol(alphaStar) * toCol(Z))
            if not self.rbf and not self.polynomial:
                self.wStar = wStar
            else:
                self.alphaStar = toRow(alphaStar)

            self.score_test_data()

        self.allScores = np.hstack(self.scores)[0]
        self.groundTrouthLabels = np.hstack(self.groundTrouthLabels)

        # print(f"allScores: {self.allScores}")
        # print(f"groundTrouthLabels: {self.groundTrouthLabels}")
        return self.allScores
    
    def score_test_data(self):
        # self.testSetExtended = np.vstack([self.testSet, np.ones((1, self.testSet.shape[1]))])
        if not self.rbf and not self.polynomial:
            self.scores.append(np.dot(self.wStar.T, self.testSetExtended))
            self.groundTrouthLabels.append(self.testSetLabel)
        else:
            Z = np.zeros(self.trainSetLabel.shape)
            Z[self.trainSetLabel == 1] = 1
            Z[self.trainSetLabel == 0] = -1
            if self.rbf:
                rbfKernel = self.rbf_kernel(train=False)
                self.scores.append(np.dot(self.alphaStar * Z, rbfKernel))
                # self.scores.append(score)
                self.groundTrouthLabels.append(self.testSetLabel)
            else:
                ployKernel = self.polynominal_kernel(train=False)
                self.scores.append(np.dot(self.alphaStar * Z, ployKernel))
                # self.scores.append(score)
                np.append(self.groundTrouthLabels, self.testSetLabel)

    def rbf_kernel(self, train=True):
        # self.testSetExtended = np.vstack([self.testSet, np.ones((1, self.testSet.shape[1]))])
        if train:
            dis = toCol((self.trainSetExtended ** 2).sum(0)) + toRow((self.trainSetExtended ** 2).sum(0)) - 2 * np.dot(self.trainSetExtended.T, self.trainSetExtended)
            rbfKernel = np.exp(-self.gamma * dis)
            return rbfKernel
        else:
            dis = toCol((self.trainSetExtended ** 2).sum(0)) + toRow((self.testSetExtended ** 2).sum(0)) - 2 * np.dot(self.trainSetExtended.T, self.testSetExtended)
            rbfKernel = np.exp(-self.gamma * dis)
            return rbfKernel

    def polynominal_kernel(self, train=True):
        if train:
            return (np.dot(self.trainSetExtended.T, self.trainSetExtended) + self.polynomial_c) ** self.d
        else:
            return (np.dot(self.trainSetExtended.T, self.testSetExtended) + self.polynomial_c) ** self.d