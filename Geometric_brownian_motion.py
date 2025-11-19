import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm_path(S0=100, mu=0.08, sigma=0.2, T=1.0, n_steps=252):
    dt = T / n_steps
    Z = np.random.normal(0.0, 1.0, size=n_steps)
    S = np.zeros(n_steps + 1)
    S[0] = S0
    for t in range(1, n_steps + 1):
        S[t] = S[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1]
        )
    time_grid = np.linspace(0, T, n_steps + 1)
    return time_grid, S

# Single path
t_gbm, S_gbm = simulate_gbm_path()

plt.plot(t_gbm, S_gbm)
plt.title("GBM Stock Price Path (1 year)")
plt.xlabel("Time (years)")
plt.ylabel("S_t")
plt.show()

# Multiple paths
def simulate_gbm_paths(S0=100, mu=0.08, sigma=0.2, T=1.0, n_steps=252, n_paths=20):
    dt = T / n_steps
    time_grid = np.linspace(0, T, n_steps + 1)
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = S0
    for i in range(n_paths):
        Z = np.random.normal(0.0, 1.0, size=n_steps)
        for t in range(1, n_steps + 1):
            paths[t, i] = paths[t-1, i] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1]
            )
    return time_grid, paths

t_grid, S_paths = simulate_gbm_paths()

plt.plot(t_grid, S_paths)
plt.title("GBM – 20 Paths")
plt.xlabel("Time (years)")
plt.ylabel("S_t")
plt.show()
