
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (9, 4)
plt.rcParams['axes.grid'] = True

np.random.seed(42)


def simulate_random_walk(n_steps=500, sigma=1.0):
    eps = np.random.normal(loc=0.0, scale=sigma, size=n_steps)
    x = np.cumsum(eps)
    return x


def simulate_random_walk_paths(n_paths=10, n_steps=500, sigma=1.0):
    paths = np.zeros((n_steps, n_paths))
    for i in range(n_paths):
        paths[:, i] = simulate_random_walk(n_steps=n_steps, sigma=sigma)
    return paths


def simulate_brownian_motion(T=1.0, n_steps=252):
    dt = T / n_steps
    increments = np.random.normal(0.0, np.sqrt(dt), size=n_steps)
    W = np.cumsum(increments)
    t_grid = np.linspace(dt, T, n_steps)
    return t_grid, W


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


def simulate_ou_process(X0=0.0, mu=0.0, theta=1.5, sigma=0.3, T=1.0, n_steps=252):
    dt = T / n_steps
    X = np.zeros(n_steps + 1)
    X[0] = X0
    for t in range(1, n_steps + 1):
        dX = theta * (mu - X[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        X[t] = X[t-1] + dX
    time_grid = np.linspace(0, T, n_steps + 1)
    return time_grid, X


def gbm_monte_carlo(S0=100, mu=0.08, sigma=0.2, T=1.0, n_steps=252, n_paths=1000):
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


def main():
    print("Week 3 – Stochastic Processes Demo")

    rw = simulate_random_walk()
    plt.plot(rw)
    plt.title("Random Walk")
    plt.xlabel("Step")
    plt.ylabel("X_t")
    plt.show()

    t, W = simulate_brownian_motion()
    plt.plot(t, W)
    plt.title("Brownian Motion $W_t$")
    plt.xlabel("Time (years)")
    plt.ylabel("W_t")
    plt.show()

    t_gbm, S_gbm = simulate_gbm_path()
    plt.plot(t_gbm, S_gbm)
    plt.title("GBM Stock Price Path")
    plt.xlabel("Time (years)")
    plt.ylabel("S_t")
    plt.show()

    t_ou, X_ou = simulate_ou_process(X0=1.0)
    plt.plot(t_ou, X_ou)
    plt.axhline(0.0, linestyle="--")
    plt.title("OU Mean-Reverting Process")
    plt.xlabel("Time (years)")
    plt.ylabel("X_t")
    plt.show()

    _, S_mc_paths = gbm_monte_carlo()
    S_T = S_mc_paths[-1, :]
    plt.hist(S_T, bins=40, density=True)
    plt.title("GBM Monte Carlo – Final Prices Distribution")
    plt.xlabel("S_T")
    plt.ylabel("Density")
    plt.show()


if __name__ == "__main__":
    main()
