import numpy as np
from abc import abstractmethod


class BootstrapEstimator:
    """ A base class for various block bootstrap methods """

    def __init__(self, X, round=True):
        """
        Parameters
        ----------
        X : array-like
            Array of data of interest
        round : bool (optional)
            Whether or not to round the optimal block length to the nearest integer. True for all except the stationary bootstrap.
        """"
        X = np.ravel(X)
        self.N = len(X)
        self.mean = np.mean(X)
        self.b = self.get_opt_block_length(X, round)
        self.var = self.estimate_variance(X)

    def get_opt_block_length(self, X, round):
        """
        Automatic optimal block length estimation based on work by Politis & White (2001, c. 2007). Adapted from MATLAB
        script written by Andrew Patton. We have extended the code to also function for estimating the optimal block
        length for the tapered block bootstrap.

        Parameters
        ----------
        X : numpy.ndarray
            A 1-dimensional numpy array containing the data in question
        round : bool (Optional; default=True)
            Dictates whether or not to round the final block length (Bstar) to the nearest integer before returning

        Returns
        -------
        Bstar : int, (float)
            Estimate of the optimal block length for the given bootstrap method and data. Returned as a float if
            round=False, rounded to nearest int if round=True. Returning as int is the default behavior.
        """
        KN = max([5, np.log10(self.N) ** 0.5])  # from Politis & White (2003) (see footnote 3); see also Politis (2001)
        mmax = int(np.ceil(self.N ** 0.5) + KN)  # max number of significant samples in each record; adds KN based on recommendation of Politis (2003)

        # max block length to consider; This was sqrt(n) in a previous implementation but was changed without justification
        # or reference explaining why this would be better.
        Bmax = np.ceil(np.min([3 * self.N ** 0.5, self.N / 3]))  # Updated Dec. 2007; NOTE: no justification/reference in original script; originally sqrt(N)

        c = 2  # from Politis & White (2003)

        record = X.reshape((-1, 1))  # gets the i-th record as a column vector

        # STEP ONE: Determine mhat, the largest lag for which the autocorrelation is still significant.
        temp = self.mlag(record, mmax)  # creates a matrix of lagged X;
        temp = temp[mmax:, :]  # only keep rows mmax and greater; this cuts out all of the rows from the matrix created by mlag() that had zeros in them (in the upper-triangular portion of the matrix)
        temp = np.hstack((record[mmax:], temp)).T  # prepends the original X from mmax onward to the lag matrix
        # At this point, going up temp's columns is going down X, where the last entry of i-th column in temp is the
        # i-th element in X. This is equivalent to np.hstack((X[i:mmax+i+1][::-1] for i in range(mmax))) but
        # handles the case of multivariate observiations (see the definition of mlag). Note that there's probably a
        # faster way to do this with numpy.roll() or something, but it alread takes so little time relative to anything
        # else (loading X from disk, SB or TBB methods, etc) that's it's a useless optimization.
        temp = np.corrcoef(temp)  # correlation coefficient matrix
        temp = temp[1:, 0]  # keep first column of matrix from second row down; corresponds to autocorrelations from lag=1 to lag=mmax
        if np.any(np.isnan(temp)):  # TODO: How should we actually deal with these cases?
            return np.nan

        # Following the empirical rule suggested by Politis ("Adaptive Bandwidth Choice", 2003; see Remark 2.3), to use
        # c = 2 and KN = 5 (set at beginning of script), we determine which autocorrelation terms are significant.
        # TODO: check sensitivity to choices of c, KN and raise warning if applicable (see advice of Politis & White (2003))
        temp2 = np.hstack((self.mlag(temp.reshape((-1, 1)), KN).T, temp[-KN:].reshape((-1, 1))))  # vectors of autocorrelations from lag mhat to lag mhat + KN
        temp2 = temp2[:, KN:]  # the first KN columns, as they have some empty cells
        temp2 = abs(temp2) < c * (np.log10(self.N) / self.N) ** 0.5  # checks which values are less than the critical value
        temp2 = np.sum(temp2, axis=0).reshape((-1, 1))  # counts the number of insignificant autocorrelations
        temp3 = np.hstack((np.arange(1, len(temp2) + 1).reshape((-1, 1)), temp2))  #
        inds = np.argwhere(temp2.ravel() == KN).ravel()
        temp3 = temp3[inds]  # selects rows where ALL KN autocorrelations are insignificant

        if temp3.size == 0:  # no collection of KN autocorrelations were all insignificant, so choose the largest significant lag
            mhat = max(np.argwhere(abs(temp) > c * (np.log10(self.N) / self.N) ** 0.5))[0]  # typical for high sampling rates (i.e. highly-correlated data); CORRECTED 6/18/2020
        else:  # if at least one collection of KN autocorrelations is all significant, choose the smallest m
            mhat = temp3[0, 0]

        M = min([2 * mhat, mmax])

        # STEP TWO: computing the inputs to the function for Bstar
        kk = np.arange(-M, M + 1)

        if M > 0:  # There's at least one significant autocorrelation term
            temp = self.mlag(record, M)
            temp = temp[M:, :]
            temp = np.cov(np.hstack((record[M:], temp)).T)
            acv = temp[:, 0].reshape((-1, 1))
            acv2 = np.hstack((-np.arange(1, M + 1).reshape((-1, 1)), acv[1:]))
            if len(acv2) > 1:
                acv2 = acv2[acv2[:, 0].argsort()]
            acv = np.vstack((acv2[:, 1].reshape((-1, 1)), acv))  # this is \hat{R}

            # See block length selector and tapered block bootstrap papers. These formulae are straight from those papers.
            # variables with CB --> circular/moving block bootstrap
            #                SB --> stationary bootstrap
            lam_acv = self.lam(kk / M) * acv.ravel()  # lam(kk/M) * acv is performed a few times in the MATLAB script so I just do it once and store it here
            # FINAL STEP: construct optimal block length
            Bstar = self.Bstar(lam_acv, kk)
        else:  # No significant autocorrelation terms --> use a block length of 1
            Bstar = 1

        # NOTE: These lines are my addition. They set the minimum block length to 1 and round the block lengths to the
        # nearest integer value. The original single_record_comparisons_6.m file doesn't round to the nearest int, opting
        # to truncate instead. However, rounding is implemented here because that's what's advocated in Politis & White (2003),
        # though this will admittedly probably not make much difference to the final results. In Bart's MATLAB code, the
        # minimum of 1 was done in his code, not the opt_block_length script.
        if Bstar > Bmax:
            Bstar = Bmax
        elif Bstar < 1:
            Bstar = 1

        if round:
            # only makes sense as an int for all but SB, which could be either a float or an int
            Bstar = np.round(Bstar).astype(int)
            # Bstar = np.floor(Bstar).astype(int)

        return Bstar

    @staticmethod
    def mlag(x, n, init=0):
        nobs, nvar = x.shape  # nobs = number of observations/samples, nvar = number of variables for which we have samples
        xlag = np.ones((nobs,
                        nvar * n)) * init  # array of lags; shape = (number of samples, number of variables * max number of significant samples)
        icnt = 0
        for i in range(nvar):
            for j in range(n):
                xlag[j + 1:, icnt + j] = x[:nobs - j - 1, i]
            icnt = icnt + n
        return xlag

    @staticmethod
    def lam(kk):
        return (abs(kk) >= 0) * (abs(kk) < 0.5) + 2 * (1 - abs(kk)) * (abs(kk) >= 0.5) * (abs(kk) <= 1)

    @abstractmethod
    def Bstar(self, lam_acv, kk):
        pass

    @abstractmethod
    def estimate_variance(self, X, b=None):
        """ Calculates an estimate of the uncertainty of the sample mean ($\sigma^2_{\bar{X}_N}$).
        Note that this is equal to sigma^2_inf / N. """
        pass


class CircularBlockBootstrap(BootstrapEstimator):
    """ The circular block bootstrap of Politis and Romano (1991) """
    def __init__(self, X):
        super().__init__(X, round=True)

    def estimate_variance(self, X, b=None):
        if b:
            self.b = b
        Y = [*X, *X]
        XNbar = np.mean(X)
        sigma2_inf = 1 / (self.b * self.N) * np.sum([np.sum(Y[i:i + int(self.b)]) ** 2 for i in range(self.N)]) - self.b * XNbar ** 2
        # return sigma2_inf / self.N
        return sigma2_inf

    def Bstar(self, lam_acv, kk):
        Dhat = 4 / 3 * np.sum(lam_acv) ** 2
        Ghat = np.sum(abs(kk) * lam_acv)
        return (self.N * 2 * Ghat ** 2 / Dhat) ** (1 / 3)


class StationaryBootstrap(BootstrapEstimator):
    """ The stationary bootstrap of Politis and Romano (1994) """
    def __init__(self, X):
        super().__init__(X, round=False)

    def estimate_variance(self, X, b=None):
        if b:
            self.b = b

        std = np.std(X, ddof=1)
        ac = acorr(X - np.mean(X), std)
        j = np.arange(1, self.N + 1)
        weights = (1 - j / self.N) * (1 - 1 / self.b) ** j + j / self.N * (1 - 1 / self.b) ** (self.N - j)
        Tu = 1 + 2 * np.sum((weights * ac)[1:self.N])
        Tu = 1 if Tu < 0 else Tu
        # return std ** 2 / self.N * Tu
        return std ** 2 * Tu

    def Bstar(self, lam_acv, kk):
        Dhat = 2 * np.sum(lam_acv) ** 2
        Ghat = np.sum(abs(kk) * lam_acv)
        return (self.N * 2 * Ghat ** 2 / Dhat) ** (1 / 3)


class TaperedBlockBootstrap(BootstrapEstimator):
    """ The tapered block bootstrap of Paparoditis and Poitis (2001) """
    def __init__(self, X):
        super().__init__(X)

    def estimate_variance(self, X, b=None):
        if b:
            self.b = b

        if self.b == 1:
            return np.var(X)

        Q = self.N - self.b + 1

        wb_vec = np.array([self._wb(k) for k in range(1, self.b + 1)])
        wb_cumsum = np.sum(wb_vec) * np.ones(X.shape)
        cs = np.cumsum(wb_vec)
        wb_cumsum[:self.b] = cs
        wb_cumsum[-self.b + 1:] -= cs[:-1]
        t2 = np.sum(X * wb_cumsum) / Q  # doesn't depend on i so we'll save it and reuse it every iteration
        tot_sum = np.sum([(np.sum(wb_vec * X[i:i + self.b]) - t2) ** 2 for i in range(self.N - self.b + 1)])
        var = tot_sum / (Q * np.sum(wb_vec ** 2))

        # return var / self.N
        return var

    def Bstar(self, lam_acv, kk):
        Dhat = 1.099 * np.sum(lam_acv) ** 2
        Ghat = -10.9 * np.sum(kk ** 2 * lam_acv)
        return (self.N * 4 * Ghat ** 2 / Dhat) ** (1 / 5)

    def _wb(self, t):
        return self._w((t - 0.5) / self.b)

    @staticmethod
    def _w(t, c=0.43):
        if 0 <= t < c:
            return t / c
        elif c <= t < 1 - c:
            return 1
        elif 1 - c <= t < 1:
            return (1 - t) / c
        else:
            return 0


class ACFTruncation:
    """ Base class for methods of approximating the integral time scale by selecting a point at which to truncate the sum of the autocorrelation function """
    def __init__(self, X):
        self.N = len(X)
        self.std = np.std(X, ddof=1)
        ac = acorr(X - np.mean(X), self.std)
        self.b = self.get_b(ac)
        Tu = 1 + 2 * np.sum(ac[1:int(self.b)])
        Tu = 1 if Tu < 0 else Tu
        self.var = self.std ** 2 / self.N * Tu

    @abstractmethod
    def get_b(self, ac):
        pass


class Minimum(ACFTruncation):
    """ Truncates the autocorrelation sum at the first minimum """
    def __init__(self, X):
        super().__init__(X)

    def get_b(self, ac):
        return np.argmin(ac) + 1


class Zero(ACFTruncation):
    """ Truncates the autocorrelation sum at the first zero """
    def __init__(self, X):
        super().__init__(X)

    def get_b(self, ac):
        return np.argwhere(ac < 0)[0, 0] + 1


class VarWrapper:
    """ Wraps the calculation of the variance for independent samples to give a consistent interface with the block bootstrap implementations here. """
    def __init__(self, X):
        self.var = np.var(X, ddof=1) / len(X)


def nonoverlapping_batch_means(std, ac):
	""" Method of non-overlapping batch means. This is used in this project to calculate the 'parent' statistics based on the full dataset.

	std: float
		Standard deviation of the 
    ac : array-like
        Values of the autocorrelation function
	"""
    Tu = 1 + 2 * np.sum(ac[1:])  # Calculation of the integral time scale
    Tu = 1 if Tu < 0 else Tu
    return std ** 2 / len(ac) * Tu


def acorr(X, std=None):
    """ Calculates the autocorrelation function (ACF) for values in X, assuming equal time spacing between indices. This method uses a
    Fast Fourier transform to calculate the ACF. This is much faster than using the definition of the autocorrelation function!
    
    Parameters
    ----------
    X : array-like
        One-dimensional array of data
    std : float (optional)
        Standard deviation of the data in X

    Returns
    -------
    acf : array-like
        The ACF of X (shape (N,))
    """
    if not std:
        std = np.std(X, ddof=1)
    N = len(X)
    fvi = np.fft.fft(X, n=2*N)
    acf = np.real(np.fft.ifft(fvi * np.conjugate(fvi))[:N])
    acf = acf / (N * std**2)
    return acf
