import numpy as np
import matplotlib.pyplot as plt

def simulate_ou_process(X0=1.0, mu=0.0, theta=1.5, sigma=0.3, T=1.0, n_steps=252):
    dt = T / n_steps
    X = np.zeros(n_steps + 1)
    X[0] = X0
    for t in range(1, n_steps + 1):
        dX = theta * (mu - X[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        X[t] = X[t-1] + dX
    time_grid = np.linspace(0, T, n_steps + 1)
    return time_grid, X

t_ou, X_ou = simulate_ou_process()

plt.plot(t_ou, X_ou)
plt.axhline(0.0, linestyle='--', label='mean')
plt.title("OU Mean-Reverting Process")
plt.xlabel("Time (years)")
plt.ylabel("X_t")
plt.legend()
plt.show()
def simulate_ou_process_paths(X0=1.0, mu=0.0, theta=1.5, sigma=0.3, T=1.0, n_steps=252, n_paths=20):
    dt = T / n_steps
    time_grid = np.linspace(0, T, n_steps + 1)
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = X0
    for i in range(n_paths):
        for t in range(1, n_steps + 1):
            dX = theta * (mu - paths[t-1, i]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            paths[t, i] = paths[t-1, i] + dX
    return time_grid, paths
t_grid, X_paths = simulate_ou_process_paths()
plt.plot(t_grid, X_paths)
plt.axhline(0.0, linestyle='--', label='mean')
plt.title("OU Mean-Reverting Process – 20 Paths")
plt.xlabel("Time (years)")
plt.ylabel("X_t")
plt.legend()
plt.show()
