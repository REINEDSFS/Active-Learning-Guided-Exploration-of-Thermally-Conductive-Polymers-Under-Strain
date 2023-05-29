# Import necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
from scipy.optimize import minimize

# Load the dataset
df = pd.read_csv('dataset2_mean uncertainty.csv')

# Define the acquisition function using Expected Improvement (EI)
def expected_improvement(x, gp, y_max):
    mean, std = gp.predict(x, return_std=True)
    z = (mean - y_max) / std
    return (mean - y_max) * norm.cdf(z) + std * norm.pdf(z)

# Normalize data
X = df['mean values'].values.reshape(-1, 1)
y = df['uncertainty'].values

# Initialize the Gaussian Process
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)

# Fit the GP model
gp.fit(X, y)

# Calculate EI for each data point
x = np.linspace(min(X), max(X), 1000).reshape(-1, 1)
ei_values = expected_improvement(x, gp, np.max(y))

# Find the top 5 polymers
top_5_indices = ei_values.argsort()[-5:][::-1]
top_5_pids = df.iloc[top_5_indices]['PID']

# Save the scores and their PID as a CSV file
df_ei = pd.DataFrame({'PID': top_5_pids, 'EI_score': ei_values[top_5_indices]})
df_ei.to_csv('EI_score.csv', index=False)

# Plot the EI scores
plt.figure(figsize=(10, 6))
plt.plot(x, ei_values, 'b-', label='Expected Improvement')
plt.scatter(X[top_5_indices], ei_values[top_5_indices], color='r', label='Top 5 Polymers')
plt.xlabel('Mean Values')
plt.ylabel('EI Score')
plt.title('Expected Improvement Score for Each Polymer')
plt.legend()
plt.show()
