import os
import math
import numpy as np
from utils import import_lightcurve
from scipy.optimize import curve_fit
from astropy.table import Table
from scipy.stats import skewnorm


"""Building the synthetic data for the training set"""


### some parameter space
# depth = 10 ** np.random.uniform(-4, -1, 1)[0] # range from 0.0001 to 0.1
# skewness = np.random.uniform(0, 30, 0.001)[0] # range from 0 to 30


def gauss(t, A, t0, sigma):
    """
    Creates a Gaussian function.

    Parameters:
        :t (float or array): Time or array of times at which to evaluate the Gaussian function.
        :A (float): Amplitude of the Gaussian peak.
        :t0 (float): Mean or centre of the Gaussian distribution.
        :sigma (float): Standard deviation or width of the Gaussian distribution.

    Returns:
        float or array: Value of the Gaussian function at the given time(s).


    Notes:
        To maintain asymmetry in the exocomet direction, always have sigma tail > sigma.
    """

    return abs(A) * np.exp(-((t - t0) ** 2) / (2 * sigma**2))


def comet_curve(t, A, t0, sigma=3.28541476e-01, tail=3.40346173e-01):
    ### add the Beta Pic parameters
    """
    Creates an exocomets light curve model.

    Notes: These are the Beta Pic parameters:
        - A = 8.84860547e-04, t0 = 1.48614591e+03, sigma = 3.28541476e-01, alpha (skewness) = 1.43857307e+00, tail = 3.40346173e-01

    Parameters:
        t (array): Independent variable (time) values.
        A (float): Amplitude of the Gaussian curve.
        t0 (float): Mean (centre) of the Gaussian curve.
        sigma (float): Standard deviation of the Gaussian curve.
        tail (float): Tail parameter controlling decay rate after t0.

    Returns:
        array: The computed values of the asymmetric Gaussian."""

    x = np.zeros(len(t))
    for i in range(len(t)):
        if t[i] < t0:
            x[i] = gauss(t[i], A, t0, sigma)
        else:
            x[i] = A * math.exp(-abs(t[i] - t0) / tail)

    return x


def comet_curve_fit(x, y):
    """Fits the exocomet light curve model to the data."""
    # Initial parameters guess
    # x = time
    # y = flux
    i = np.argmax(y)

    width = x[-1] - x[0]

    params_init = [y[i], x[i], width / 3, width / 3]

    params_bounds = [[0, x[0], 0, 0], [np.inf, x[-1], width / 2, width / 2]]
    params, cov = curve_fit(comet_curve, x, y, params_init, bounds=params_bounds)

    return params, cov

def skewed_gaussian(x, alpha=1, t0=1496.5, sigma=1, depth=0.001):
    """Creates a skewed Gaussian model transit.

    Parameters:
        x: Time array.
        alpha: Skewness parameter (0 for symmetric).
        t0: Mean of the Gaussian.
        sigma: Standard deviation of the Gaussian.
        depth: Transit depth.

    Returns:
        y: The value of the skewed Gaussian at each input data point x.
    """
    pdf = skewnorm.pdf(x, alpha, loc=t0, scale=sigma)
    normalized_pdf = pdf / pdf.max()
    return 1 - (depth * normalized_pdf)


def comet_ingress(x, A, mu, sigma, shape):
    '''exponential comet ingress, `shape` controls curviness'''
    sh = shape/sigma
    norm = 1 - np.exp(-sh*sigma)
    return A/norm*(1 - np.exp(-sh*(x-mu+sigma)))

def comet_curve2(x, A, mu, sigma, tail, shape=3):
    return np.piecewise(x, [x<(mu-sigma), np.logical_and(x>=(mu-sigma), x<mu), x>=mu],
                        [0,
                         lambda t: comet_ingress(t, A, mu, sigma, shape),
                         lambda t: A*np.exp(-abs(t-mu)/tail)])


def ldecomet(t, K, beta, t0, delta_t):
    # Calculate Δ
    delta = np.where(t >= t0, beta * (t - t0), 0)
    
    # Calculate Δ'
    delta_prime = np.where(t >= t0 + delta_t, beta * (t - t0 - delta_t), 0)
    
    # Calculate the relative flux decrease
    delta_f_over_f = K * (np.exp(-delta) - np.exp(-delta_prime))
    
    return delta_f_over_f