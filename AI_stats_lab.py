"""
AI Mathematical Tools – Probability & Random Variables

Instructions:
- Implement ALL functions.
- Do NOT change function names or signatures.
- Do NOT print inside functions.
- You may use: math, numpy, matplotlib.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Part 1 — Probability Foundations
# ============================================================

def probability_union(PA, PB, PAB):
    """
    P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
    """
    # Check numeric
    for p in (PA, PB, PAB):
        if type(p) not in (int, float):
            raise TypeError("All inputs must be numeric")
    
    # Check range
    for p in (PA, PB, PAB):
        if not 0 <= p <= 1:
            raise ValueError("All probabilities must be between 0 and 1")
    
    # Check logical consistency
    if PAB > min(PA, PB):
        raise ValueError("P(A ∩ B) cannot exceed min(PA, PB)")
    
    # Compute union
    return PA + PB - PAB


def conditional_probability(PAB, PB):
    """
    P(A|B) = P(A ∩ B) / P(B)
    """
     # Check probabilities are in [0,1]
    if not (0 <= PAB <= 1 and 0 <= PB <= 1):
        raise ValueError("Probabilities must be between 0 and 1")

    # PB not zero for conditional probability
    if PB == 0:
        raise ValueError("P(B) cannot be zero for conditional probability")
        
    # Logical consistency
    if PAB > PB:
        raise ValueError("P(A ∩ B) cannot be greater than P(B)")
    
    return PAB / PB


def are_independent(PA, PB, PAB, tol=1e-9):
    """
    True if:
        |P(A ∩ B) - P(A)P(B)| < tol
    """
    # Check valid probability ranges
    if not (0 <= PA <= 1 and 0 <= PB <= 1 and 0 <= PAB <= 1):
        raise ValueError("Probabilities must be between 0 and 1")
    
    # Logical consistency
    if PAB > min(PA, PB):
        raise ValueError("P(A ∩ B) cannot be greater than min(P(A), P(B))")
    
    # Independence condition
    return abs(PAB - (PA * PB)) < tol


def bayes_rule(PBA, PA, PB):
    """
    P(A|B) = P(B|A)P(A) / P(B)
    """
    # Check valid probability ranges
    if not (0 <= PBA <= 1 and 0 <= PA <= 1 and 0 <= PB <= 1):
        raise ValueError("Probabilities must be between 0 and 1")
    
    # PB must not be zero
    if PB == 0:
        raise ValueError("P(B) cannot be zero for Bayes rule")
    
    # Logical consistency: P(B|A)*P(A) <= P(B)
    if PBA * PA > PB:
        raise ValueError("P(B|A)*P(A) cannot exceed P(B)")
    
    # Bayes probability
    return (PBA * PA) / PB


# ============================================================
# Part 2 — Bernoulli Distribution
# ============================================================

def bernoulli_pmf(x, theta):
    """
    f(x, theta) = theta^x (1-theta)^(1-x)
    """
    # Validate theta
    if not (0 <= theta <= 1):
        raise ValueError("Theta must be between 0 and 1")
    
    # Validate x
    if x not in (0, 1):
        raise ValueError("x must be 0 or 1 for Bernoulli PMF")
    
    # Compute PMF
    return (theta ** x) * ((1 - theta) ** (1 - x))


def bernoulli_theta_analysis(theta_values):
    """
    Returns:
        (theta, P0, P1, is_symmetric)
    """
    result = []
    
    for theta in theta_values:
        # Reject non-numeric and booleans
        if type(theta) not in (int, float):
            raise TypeError("theta must be a numeric value (int or float)")
        
        if not 0 <= theta <= 1:
            raise ValueError("theta must be between 0 and 1")
        
        P1 = theta
        P0 = 1 - theta
        result.append((theta, P0, P1, abs(P0 - P1) < 1e-9))
    
    return result


# ============================================================
# Part 3 — Normal Distribution
# ============================================================

def normal_pdf(x, mu, sigma):
    """
    Normal PDF:
        1/(sqrt(2π)σ) * exp(-(x-μ)^2 / (2σ^2))
    """
    # Check types
    if type(x) not in (int, float):
        raise TypeError("x must be numeric")
    if type(mu) not in (int, float):
        raise TypeError("mu must be numeric")
    
    # Check sigma > 0
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    
    # Compute PDF
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def normal_histogram_analysis(mu_values,
                              sigma_values,
                              n_samples=10000,
                              bins=30):
    """
    For each (mu, sigma):

    Return:
        (
            mu,
            sigma,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        )
    """
    results = []

    for mu, sigma in zip(mu_values, sigma_values):

        samples = np.random.normal(mu, sigma, n_samples)

        sample_mean = np.mean(samples)
        sample_variance = np.var(samples)

        theoretical_mean = mu
        theoretical_variance = sigma ** 2

        mean_error = abs(sample_mean - theoretical_mean)
        variance_error = abs(sample_variance - theoretical_variance)

        results.append((
            mu,
            sigma,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        ))

    return results


# ============================================================
# Part 4 — Uniform Distribution
# ============================================================

def uniform_mean(a, b):
    """
    (a + b) / 2
    """
    if type(a) not in (int, float) or type(b) not in (int, float):
        raise TypeError("a and b must be numeric")
    if a > b:
        raise ValueError("a must be less than or equal to b")
    return (a + b) / 2


def uniform_variance(a, b):
    """
    (b - a)^2 / 12
    """
    if type(a) not in (int, float) or type(b) not in (int, float):
        raise TypeError("a and b must be numeric")
    if a > b:
        raise ValueError("a must be less than or equal to b")
    return ((b - a) ** 2) / 12


def uniform_histogram_analysis(a_values,
                               b_values,
                               n_samples=10000,
                               bins=30):
    """
    For each (a, b):

    Return:
        (
            a,
            b,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        )
    """
    results = []

    for a, b in zip(a_values, b_values):

        samples = np.random.uniform(a, b, n_samples)

        sample_mean = np.mean(samples)
        sample_variance = np.var(samples)

        theoretical_mean = uniform_mean(a, b)
        theoretical_variance = uniform_variance(a, b)

        mean_error = abs(sample_mean - theoretical_mean)
        variance_error = abs(sample_variance - theoretical_variance)

        results.append((
            a,
            b,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        ))

    return results


if __name__ == "__main__":
    print("Implement all required functions.")
