import numpy as np
import matplotlib.pyplot as plt

def gbm_monte_carlo(S0=100, mu=0.08, sigma=0.2,
                    T=1.0, n_steps=252, n_paths=1000):
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

t_mc, S_mc_paths = gbm_monte_carlo()
S_T = S_mc_paths[-1, :]

plt.hist(S_T, bins=40, density=True)
plt.title("GBM Monte Carlo – Distribution of $S_T$")
plt.xlabel("S_T")
plt.ylabel("Density")
plt.show()

mean_ST = np.mean(S_T)
std_ST = np.std(S_T)
prob_below_90 = np.mean(S_T < 90)
prob_above_120 = np.mean(S_T > 120)

log_returns = np.log(S_T / 100.0)
expected_return = np.mean(np.exp(log_returns) - 1)
vol_of_returns = np.std(log_returns)

print(f"Mean S_T: {mean_ST:.2f}")
print(f"Std(S_T): {std_ST:.2f}")
print(f"P(S_T < 90):  {prob_below_90:.3f}")
print(f"P(S_T > 120): {prob_above_120:.3f}")
print(f"Approx expected return: {expected_return:.2%}")
print(f"Vol of log-returns: {vol_of_returns:.2%}")
