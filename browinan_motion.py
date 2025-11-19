import numpy as np
import matplotlib.pyplot as plt
def simulate_brownian_motion(T=1.0, n_steps=252):
    dt = T / n_steps
    increments = np.random.normal(0.0, np.sqrt(dt), size=n_steps)
    W = np.cumsum(increments)
    t = np.linspace(dt, T, n_steps)
    return t, W

t_bm, W = simulate_brownian_motion(T=1.0, n_steps=252)

plt.plot(t_bm, W)
plt.title("Brownian Motion $W_t$")
plt.xlabel("Time (years)")
plt.ylabel("W_t")
plt.show()
