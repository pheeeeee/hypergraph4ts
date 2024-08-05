



import numpy as np
import matplotlib.pyplot as plt

def henon_like_map(a, b, initial_conditions, num_steps):
    """
    Simulates a simplified, one-dimensional version of the Henon map.
    
    Parameters:
        a (float): The coefficient for the x^2 term.
        b (float): The coefficient for the lagged x term.
        initial_conditions (list of float): Initial values [x0, x1].
        num_steps (int): Number of steps to simulate.
    
    Returns:
        np.array: The generated time series.
    """
    x = np.zeros(num_steps)
    x[0], x[1] = initial_conditions
    
    for i in range(2, num_steps):
        x[i] = 1 - a * x[i-1]**2 + b * x[i-2]
    
    return x

# Parameters
a = 1.4
b = 0.3
initial_conditions = [0.1, -0.1]  # [x0, x1]
num_steps = 100  # Number of steps to simulate

# Generate the time series
x_values = henon_like_map(a, b, initial_conditions, num_steps)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, '.-')
plt.title('Henon like Map')
plt.xlabel('Time Step')
plt.ylabel('Population Ratio $x_n$')
plt.grid(True)
plt.show()







def generalized_dynamical_system(c, a_coeffs, b_coeffs, initial_conditions, num_steps):
    """
    Simulates a generalized one-dimensional dynamical system with specified coefficients for
    linear and non-linear (quadratic) terms.
    
    Parameters:
        c (float): Constant term in the dynamical equation.
        a_coeffs (list of float): Coefficients for the quadratic terms.
        b_coeffs (list of float): Coefficients for the linear terms.
        initial_conditions (list of float): Initial values; length should match the maximum of len(a_coeffs) and len(b_coeffs).
        num_steps (int): Number of steps to simulate.
    
    Returns:
        np.array: The generated time series.
    """
    max_lags = max(len(a_coeffs), len(b_coeffs))
    if len(initial_conditions) < max_lags:
        raise ValueError("Initial conditions must at least cover the maximum lag required by the coefficients.")
    
    x = np.zeros(num_steps + max_lags)
    x[:max_lags] = initial_conditions[:max_lags]
    
    for i in range(max_lags, num_steps + max_lags):
        x[i] = c - sum(a_coeffs[j] * x[i-j-1]**2 for j in range(len(a_coeffs))) + sum(b_coeffs[j] * x[i-j-1] for j in range(len(b_coeffs)))
    
    return x[max_lags:]  # Exclude the initial conditions part in the returned result

# Parameters
c = 0.5
a_coeffs = [0.1, 0.05]  # Coefficients for quadratic terms
b_coeffs = [-0.3, 0.2, 0.1]  # Coefficients for linear terms
initial_conditions = [0.1, -0.1, 0.05]
num_steps = 100  # Number of steps to simulate

# Generate the time series
x_values = generalized_dynamical_system(c, a_coeffs, b_coeffs, initial_conditions, num_steps)












def extended_logistic_map(r, a_coeffs, initial_conditions, num_steps):
    """
    Simulates an extended logistic map with multiple lagged terms.
    
    Parameters:
        r (float): The logistic growth rate.
        a_coeffs (list of float): Coefficients for the lagged terms.
        initial_conditions (list of float): Initial conditions; length should cover the lags needed.
        num_steps (int): Number of steps to simulate.
    
    Returns:
        np.array: The generated time series.
    """
    max_lags = len(a_coeffs)
    if len(initial_conditions) < max_lags:
        raise ValueError("Not enough initial conditions for the specified number of lags.")
    
    x = np.zeros(num_steps + max_lags)
    x[:max_lags] = initial_conditions
    
    for i in range(max_lags, num_steps + max_lags):
        x[i] = r * x[i-1] * (1 - x[i-1]) + sum(a_coeffs[j] * x[i-j-1] for j in range(max_lags))
    
    return x[max_lags:]  # Skip the initial condition part in the output

# Parameters
r = 3.7  # Logistic growth rate
a_coeffs = [0.05, -0.02, 0.01]  # Coefficients for previous lags
initial_conditions = [0.1, 0.2, 0.3]
num_steps = 100

# Generate the time series
x_values = extended_logistic_map(r, a_coeffs, initial_conditions, num_steps)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, '.-')
plt.title('Extended Logistic Map with Multiple Lags')
plt.xlabel('Time Step')
plt.ylabel('Population Ratio $x_n$')
plt.grid(True)
plt.show()







import numpy as np
import matplotlib.pyplot as plt

def mackey_glass(beta, gamma, n, tau, initial_conditions, num_steps):
    """
    Simulates the Mackey-Glass delay differential equation in a discrete setting.
    
    Parameters:
        beta (float): Production rate parameter.
        gamma (float): Decay rate parameter.
        n (float): Exponent in the production term.
        tau (int): Delay term, must be a positive integer.
        initial_conditions (list of float): Initial conditions, length must be at least tau + 1.
        num_steps (int): Number of steps to simulate.
    
    Returns:
        np.array: The generated time series.
    """
    
    x = np.zeros(num_steps + tau)
    x[:tau] = initial_conditions
    
    for t in range(tau, num_steps + tau):
        x[t] = x[t-1] + (beta * x[t-tau] / (1 + x[t-tau]**n) - gamma * x[t-1])
    
    return x[tau:]  # Return the series excluding the initial condition part used for delay

# Parameters
beta = 0.2
gamma = 0.1
n = 10
tau = 5  # Delay term
initial_conditions = [0.9] * (tau)  # Uniform initial conditions
num_steps = 1000

# Generate the time series
x_values = mackey_glass(beta, gamma, n, tau, initial_conditions, num_steps)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values)
plt.title('Mackey-Glass Equation Simulation')
plt.xlabel('Time Step')
plt.ylabel('x(t)')
plt.grid(True)
plt.show()






## Regression
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Generate synthetic data
np.random.seed(0)
n = 100
x = np.random.randn(n)
# Simulate some dependencies
y = 1.5 * x + np.random.randn(n) * 0.5 + 3
data = pd.DataFrame({'Y': y, 'X': x})


# Define the number of lags
num_lags = 3
for lag in range(1, num_lags + 1):
    data[f'X_lag_{lag}'] = data['X'].shift(lag)

# Drop rows with NaN values resulting from lagging
data = data.dropna()


X = data.drop('Y', axis=1)
y = data['Y']

# Add a constant term to allow statsmodels to fit an intercept
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())



import pickle

# Path to your pickle file
pickle_file = '/Users/piprober/Desktop/g4tsempirical/real/experiments/bettis.pkl'

# Open the pickle file in read-binary mode
with open(pickle_file, 'rb') as file:
    # Load the object from the file
    data = pickle.load(file)

# Now 'data' contains the object that was pickled
print(data)









import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

# Suppress warnings
warnings.simplefilter('ignore', ConvergenceWarning)

# Assuming your time series data is in a pandas DataFrame `df` with a column named 'value'
# df = pd.read_csv('your_time_series_data.csv')  # Uncomment and modify this line to load your data

# Example data (replace this with your actual data)
np.random.seed(0)
data = np.random.randn(100)
df = pd.DataFrame(data, columns=['value'])

# Fit an autoregressive model
model = AutoReg(df['value'], lags=20, old_names=False).fit()

# Get the summary of the model
summary = model.summary()
print(summary)

# Determine the significant lags
significant_lags = model.pvalues[model.pvalues < 0.05].index.tolist()

# Filter out the intercept if it's included
significant_lags = [lag for lag in significant_lags if 'L' in lag]

print(f"Significant lags at 0.05 significance level: {significant_lags}")