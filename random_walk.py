import numpy as np
import matplotlib.pyplot as plt
def simulate_random_walk(n_steps=500, sigma=1.0):
    eps = np.random.normal(0.0, sigma, n_steps)
    return np.cumsum(eps)

# Single path
rw = simulate_random_walk(n_steps=500, sigma=1.0)

plt.plot(rw)
plt.title("Random Walk – Single Path")
plt.xlabel("Step")
plt.ylabel("X_t")
plt.show()

# Multiple paths
n_paths = 10
paths = np.zeros((500, n_paths))
for i in range(n_paths):
    paths[:, i] = simulate_random_walk(n_steps=500, sigma=1.0)

plt.plot(paths)
plt.title("Random Walk – 10 Paths")
plt.xlabel("Step")
plt.ylabel("X_t")
plt.show()
