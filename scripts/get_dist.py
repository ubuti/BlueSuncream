import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, poisson, t, chi2, uniform, binom, gamma, beta

def plot_continuous_distribution(func, name='distribution', x_range=(-10, 10)):
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = func(x)
    integral = np.trapz(y, x)
    y /= integral  # Normalize the distribution
    plt.plot(x, y, color='blue')
    plt.fill_between(x, y, color='blue', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.tight_layout()
    plt.savefig(f'../images/plot_{name}_distribution.png')
    plt.show()

def plot_discrete_distribution(x_values, probs, name='distribution'):
    total_prob = np.sum(probs)
    probs /= total_prob  # Normalize the probabilities
    plt.stem(x_values, probs, basefmt=" ")
    plt.xlabel('x')
    plt.ylabel('Probability Mass')
    plt.tight_layout()
    plt.savefig(f'../images/plot_{name}_distribution.png')
    plt.show()

# Example usage:

# Normal distribution
plot_continuous_distribution(lambda x: norm.pdf(x), 'normal')

# Mixed normal distribution
def mixed_normal(x):
    return 0.5 * norm.pdf(x) + 0.5 * norm.pdf(x, loc=3)

plot_continuous_distribution(mixed_normal, 'mixed_normal')

# Exponential distribution with lambda = 1
plot_continuous_distribution(lambda x: expon.pdf(x), 'exponential', x_range=(0, 10))

# Poisson distribution with lambda = 3
x_poisson = np.arange(0, 15)
probs_poisson = poisson.pmf(x_poisson, mu=3)
plot_discrete_distribution(x_poisson, probs_poisson, 'poisson')

# Overlay multiple Student's t-distributions with different degrees of freedom
def plot_t_distributions_overlay():
    x = np.linspace(-10, 10, 1000)
    plt.figure()
    for df in [1, 2, 5, 10, 20]:
        y = t.pdf(x, df)
        plt.plot(x, y, label=f'df={df}')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../images/plot_t_distributions_overlayed.png')
    plt.show()

plot_t_distributions_overlay()

# Uniform distribution between -5 and 5
plot_continuous_distribution(lambda x: uniform.pdf(x, loc=-5, scale=10), 'uniform', x_range=(-10, 10))

# Chi-squared distribution with 2-7 degrees of freedom (overlaid)
def plot_chi_squared_overlay():
    x = np.linspace(0, 20, 1000)
    plt.figure()
    for df in range(2, 8):
        y = chi2.pdf(x, df)
        plt.plot(x, y, label=f'df={df}')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../images/plot_chi_squared_overlayed.png')
    plt.show()

plot_chi_squared_overlay()

# Bernoulli distribution with p = 0.7
def plot_bernoulli_distribution(p):
    x_values = [0, 1]
    probs = [1 - p, p]
    plot_discrete_distribution(x_values, probs, 'bernoulli')

plot_bernoulli_distribution(0.5)

# Binomial distribution with n=15 and p=0.3
def plot_binomial_distribution(n, p):
    x_values = np.arange(0, n+1)
    probs = binom.pmf(x_values, n, p)
    plot_discrete_distribution(x_values, probs, 'binomial')

plot_binomial_distribution(15, 0.3)


# Gamma distribution with shape k=2 and scale theta=2
plot_continuous_distribution(lambda x: gamma.pdf(x, a=2, scale=2), 'gamma', x_range=(0, 20))

# Beta distribution with alpha=2 and beta=5
plot_continuous_distribution(lambda x: beta.pdf(x, a=2, b=5), 'beta', x_range=(0, 1))