# GP based Bayesian Active Learning Library for User Qualitative Feedback
from scipy.optimize import minimize

from Optimization.util import *
from scipy.linalg import inv
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

#np.random.seed(100)

class QualitativeFeedbackGP:
    # This GP model uses Ordinal Classifications with Pair-wise Preferences
    # It uses two parameter inputs for each trial, two classifications and one pair-wise preference
    def __init__(self, dimension=1, theta=0.1, noise_level=None, thresholds=None):

        if noise_level is None:
            noise_level = [1., 1.]
        if thresholds is None:
            thresholds=[-np.infty,0,np.infty]

        self.thresholds = thresholds
        self.listQueries = []           # list of queries
        self.K = np.zeros((2, 2))       # Covariance matrix for our queries
        self.Kinv = np.zeros((2, 2))    # inverse of the covariance matrix
        self.fqmean = np.array([])                 # posterior mean for the queries
        self.theta = theta              # RBF kernel hyperparameter for smoothness
        self.W = np.zeros((2, 2))       # hessian at the queries
        self.B = np.zeros((2,2))
        self.noisePw = noise_level[0]   # Pair-wise Preference noise
        self.noiseOrd = noise_level[1]  # Ordinal Classification noise

        self.dim = dimension            # number of input features
        self.classificationList = np.array([])

    def updateParameters(self, query, answer, classification):
        self.UploadQuery(query, answer, classification)
        self.Update()

    def UploadQuery(self,query,answer,classification):
        self.listQueries.append([query[0], query[1], answer])
        self.classificationList = np.append(self.classificationList, classification).astype(int)

    def Update(self):
        # Updates GP Model without adding new parameters
        self.covK()                                                         # Updates Prior Cov. Matrix, K
        self.fqmean = self.meanmode()                                       # Calculates meanmode, fhat
        self.W = self.hessian()                                             # Calclates negative hessian, W
        self.B = np.identity(len(self.fqmean)) + self.K @ self.W            # I +WK
        self.Binv = np.linalg.solve(self.B, np.identity(self.B.shape[0]))   #
        self.postvar = self.K @ self.Binv

    def kernel(self, xa, xb):
        sigma=1
        rbf = sigma**2 * (np.exp(-self.theta * np.linalg.norm(np.array(xa) - np.array(xb)) ** 2))
        return rbf

    def meanmode(self):  # find the posterior means for the queries
        n = len(self.listQueries)
        Kinv = self.Kinv
        listResults = np.array(self.listQueries,dtype=object)[:, 2]

        # thresholds of ordinal classification
        tr = np.array([self.thresholds[i] for i in self.classificationList])
        tr_1 = np.array([self.thresholds[i - 1] for i in self.classificationList])

        def logposterior(f):
            # pairwise comparison P(f_1>f_2)=Phi((f_1 -f_2)/c_p)
            fodd = f[1::2]
            feven = f[::2]

            fint = (1 / self.noisePw) * (feven - fodd)
            res = np.multiply(fint, listResults)
            res = res.astype(dtype=np.float64)
            res = norm.cdf(res)
            res = np.log(res)
            logPp = np.sum(res)

            # ordinal classification
            # P(b_r>f_1>b_r-1)=Phi((b_r - f)/c_o)-Phi((b_r-1 - f)/c_o)
            # alternatively, = Phi((f - b_r-1)/c_o)-Phi((f - b_r)/c_o)

            ord = norm.cdf((1 / self.noiseOrd) * (tr - f)) - norm.cdf((1 / self.noiseOrd) * (tr_1 - f))

            ord = np.log(ord)

            logPo = np.sum(ord)

            ftransp = f.reshape(-1, 1)
            logFeedback = logPo + logPp

            return -1 * (logFeedback - 2 * np.matmul(f, np.matmul(Kinv, ftransp)))

        def gradientlog(f):

            dPp = np.zeros(2 * len(self.listQueries))
            dPo = np.zeros(2 * len(self.listQueries))

            for i in range(len(self.listQueries)):
                # Pairwise

                signe = self.listQueries[i][2]
                dPp[2 * i] = self.listQueries[i][2] * (
                        phip(signe * 1 / self.noisePw * (f[2 * i] - f[2 * i + 1])) * 1 / self.noisePw) / phi(
                    signe * 1 / self.noisePw * (f[2 * i] - f[2 * i + 1]))

                dPp[2 * i + 1] = self.listQueries[i][2] * (
                        -phip(signe * 1 / self.noisePw * (f[2 * i] - f[2 * i + 1])) * 1 / self.noisePw) / phi(
                    signe * 1 / self.noisePw * (f[2 * i] - f[2 * i + 1]))

            for i in range(2 * len(self.listQueries)):
                # Ordinal
                zr = (tr[i] - f[i]) / self.noiseOrd
                zr_1 = (tr_1[i] - f[i]) / self.noiseOrd
                dPo[i] = ((phip(zr_1) - phip(zr)) * 1 / self.noiseOrd) / (phi(zr) - phi(zr_1))

            grad = dPo + dPp

            grad = grad - f @ Kinv
            return -grad

        x0 = np.zeros(2 * n)

        res = fsolve(gradientlog, x0, xtol=5e-14)
        return res #minimize(logposterior, x0=x0, jac=gradientlog).x

    def hessian(self):
        # W=WPp + WPo sum of the hessians of preference and classification terms
        n = len(self.listQueries)

        WPp = np.zeros((2 * n, 2 * n))
        WPo = np.zeros((2 * n, 2 * n))

        tr = np.array([self.thresholds[i] for i in self.classificationList])
        tr_1 = np.array([self.thresholds[i - 1] for i in self.classificationList])

        for i in range(n):
            dif = self.listQueries[i][2] * 1 / self.noisePw * (self.fqmean[2 * i] - self.fqmean[2 * i + 1])

            WPp[2 * i][2 * i] = -(1 / self.noisePw ** 2) * (phipp(dif) * phi(dif) - phip(dif) ** 2) / (phi(dif) ** 2)
            WPp[2 * i + 1][2 * i] = -WPp[2 * i][2 * i]
            WPp[2 * i][2 * i + 1] = -WPp[2 * i][2 * i]
            WPp[2 * i + 1][2 * i + 1] = WPp[2 * i][2 * i]

        for i in range(2 * n):
            zr = (tr[i] - self.fqmean[i]) / self.noiseOrd
            zr_1 = (tr_1[i] - self.fqmean[i]) / self.noiseOrd

            Po = phi(zr) - phi(zr_1)
            Pop = 1 / (self.noiseOrd) * (phip(zr_1) - phip(zr))
            Popp = 1 / (self.noiseOrd ** 2) * (phipp(zr) - phipp(zr_1))

            WPo[i][i] = -(Popp * Po - Pop ** 2) / (Po ** 2)

        return WPo + WPp

    def kt(self, xa, xb):  # covariance between xa,xb and our queries
        n = len(self.listQueries)
        return np.array([[self.kernel(xa, self.listQueries[i][j]) for i in range(n) for j in range(2)],
                         [self.kernel(xb, self.listQueries[i][j]) for i in range(n) for j in range(2)]])

    def covK(self):  # covariance matrix for all of our queries
        n = len(self.listQueries)
        self.K = np.array(
            [[self.kernel(self.listQueries[i][j], self.listQueries[l][m]) for l in range(n) for m in range(2)] for i in
             range(n) for j in range(2)])
        self.Kinv = np.linalg.solve(self.K + np.identity(2*n) * 1e-6, np.identity(self.K.shape[0]))

    def postmean(self, xa, xb):  # mean vector for two points xa and xb
        kt = self.kt(xa, xb)
        mean = kt @ (self.Kinv @ (self.fqmean))
        return mean

    def cov1pt(self, x):  # variance for 1 point
        return self.postcov(x, 0)[0][0]

    def mean1pt(self, x):
        n = len(self.listQueries)
        kt=np.array([self.kernel(x, self.listQueries[i][j]) for i in range(n) for j in range(2)])
        return  kt @ (self.Kinv @ (self.fqmean))

    def postcov(self, xa, xb):  # posterior covariance matrix for two points
        n = len(self.listQueries)
        Kt = np.array([[self.kernel(xa, xa), self.kernel(xa, xb)], [self.kernel(xb, xa), self.kernel(xb, xb)]])
        kt = self.kt(xa, xb)
        W = self.W
        K = self.K
        return Kt - kt @ inv(np.identity(2 * n) + np.matmul(W, K)) @ W @ np.transpose(kt)

    def MLE_Tune(self):
        #Hyper-Parameter Tuning with MLE,
        # -!- In Progress -!-
        # res = shgo(loglikelihood, [(0.1,20)], n=32, sampling_method='sobol')
        pass

    def loglikelihood(self,t):
        self.theta=t[0]
        self.noisePw=t[1]
        self.noiseOrd=t[2]
        self.thresholds[1]=-t[3]
        self.thresholds[2]=t[4]

        n = len(self.listQueries)
        y = self.classificationList

        self.Update()
        fhat=self.fqmean
        fhat_tr=fhat.reshape(-1, 1)
        Kinv=self.Kinv

        listResults = np.array(self.listQueries,dtype=object)[:, 2]

        # thresholds of ordinal classification
        tr = np.array([self.thresholds[i] for i in self.classificationList])
        tr_1 = np.array([self.thresholds[i - 1] for i in self.classificationList])

        def logposterior(f):
            # pairwise comparison P(f_1>f_2)=Phi((f_1 -f_2)/c_p)
            fodd = f[1::2]
            feven = f[::2]

            fint = (1 / self.noisePw) * (feven - fodd)
            res = np.multiply(fint, listResults)
            res = res.astype(dtype=np.float64)
            res = norm.cdf(res)
            res = np.log(res)
            logPp = np.sum(res)

            # ordinal classification
            # P(b_r>f_1>b_r-1)=Phi((t_r - f)/c_o)-Phi((t_r-1 - f)/c_o)
            # alternatively, = Phi((f - t_r-1)/c_o)-Phi((f - t_r)/c_o)

            ord = norm.cdf((1 / self.noiseOrd) * (tr - f)) - norm.cdf((1 / self.noiseOrd) * (tr_1 - f))
            ord = np.log(ord)
            logPo = np.sum(ord)


            logFeedback = logPo + logPp

            return logFeedback

        logp = logposterior(fhat)

        B=self.B # I +WK

        logq=-0.5*fhat@Kinv@fhat_tr-0.5*np.log(np.linalg.det(B))+logp  # log(P(q|f)P(f))

        return -logq[0]


def Plot_TwoDim(GP):
    # Plotting the two dimensional estimations of GP model

    size = 20  # resolution of the plot

    # Creating the Parameter space
    x1 = np.linspace(0, 1, size)
    x2 = np.linspace(0, 1, size)

    X1, X2 = np.meshgrid(x1, x2)
    # Finding Mean and Variance
    Mean = np.zeros((size, size))
    Var = np.zeros((size, size))

    mean_arr = np.array([])
    var_arr = np.array([])

    for i in range(size):
        for j in range(size):
            if i == 0 and j == 0:
                mean = GP.mean1pt([x1[j], x2[i]])
                mean_arr = np.array([mean])
                var_arr = np.array([GP.cov1pt([x1[j], x2[i]])])

            else:
                mean = GP.mean1pt([x1[j], x2[i]])
                mean_arr = np.vstack((mean_arr, mean))
                var_arr = np.vstack((var_arr, GP.cov1pt([x1[j], x2[i]])))

    Mean = mean_arr.reshape(size, -1)
    Var = np.sqrt(var_arr).reshape(size, -1)

    # ---------------------------------
    #  Plotting

    # PLOTTING
    fig = plt.figure(figsize=(16, 6))
    # row column
    ax11 = fig.add_subplot(2, 5, 1, projection='3d')
    ax12 = fig.add_subplot(2, 5, 2, projection='3d')

    ax11.plot_surface(X1, X2, Mean, rstride=1, cstride=1, antialiased=False,
                      cmap='coolwarm', edgecolor='none', vmin=-1, vmax=1)
    ax12.plot_surface(X1, X2, Var, rstride=1, cstride=1, antialiased=False,
                      cmap='coolwarm', edgecolor='none', vmin=0, vmax=0.7)

    plt.tight_layout()
    plt.show()
