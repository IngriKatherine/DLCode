"""
Backpropagation for a 1-S1-1 network (log-sigmoid hidden layer, linear output)
Target: g(p) = exp(-|p|) * sin(pi*p),  -2 <= p <= 2

Backprop summary (matrix form, M = 2 layers):
    Forward : a0 = p
              a^{m+1} = f^{m+1}(W^{m+1} a^m + b^{m+1})
    Backward: s^M = -2 * Fdot^M(n^M) (t - a)
              s^m = Fdot^m(n^m) (W^{m+1})^T s^{m+1}
    Update  : W^m <- W^m - alpha * s^m (a^{m-1})^T
              b^m <- b^m - alpha * s^m

mode = "sgd"   -> stochastic gradient: update after every single sample
mode = "batch" -> true gradient: average gradient over all samples, one update per epoch
"""
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- transfer functions -----------------------------
def logsig(n):
    return 1.0 / (1.0 + np.exp(-n))

def logsig_dot(a):          # derivative expressed through the output a
    return a * (1.0 - a)

def purelin(n):
    return n

def purelin_dot(a):
    return np.ones_like(a)


# ------------------------------- target function ------------------------------
def g(p):
    return np.exp(-np.abs(p)) * np.sin(np.pi * p)


# ------------------------------ network functions -----------------------------
def init_params(S1, seed=None):
    """Weights/biases uniform in [-0.5, 0.5]  (rand - 0.5)."""
    rng = np.random.default_rng(seed)
    W1 = rng.random((S1, 1)) - 0.5
    b1 = rng.random((S1, 1)) - 0.5
    W2 = rng.random((1, S1)) - 0.5
    b2 = rng.random((1, 1)) - 0.5
    return [W1, b1, W2, b2]


def forward(params, P):
    """P: (1, Q) row of inputs. Works for any number of columns (matrix ops)."""
    W1, b1, W2, b2 = params
    a0 = P
    a1 = logsig(W1 @ a0 + b1)
    a2 = purelin(W2 @ a1 + b2)
    return a0, a1, a2


def backprop_gradients(params, P, T):
    """Returns gradients averaged over the columns of P (1 column = plain SGD)."""
    W1, b1, W2, b2 = params
    Q = P.shape[1]
    a0, a1, a2 = forward(params, P)
    e = T - a2

    s2 = -2.0 * purelin_dot(a2) * e                 # (1, Q)
    s1 = logsig_dot(a1) * (W2.T @ s2)               # (S1, Q)

    dW2 = s2 @ a1.T / Q
    db2 = s2.sum(axis=1, keepdims=True) / Q
    dW1 = s1 @ a0.T / Q
    db1 = s1.sum(axis=1, keepdims=True) / Q
    return [dW1, db1, dW2, db2]


def train(S1=10, alpha=0.1, epochs=2000, mode="batch", seed=None,
          n_points=41, shuffle=True):
    """
    Train the 1-S1-1 network.
    Returns: params, sse_history (sum of squared errors per epoch), P, T
    """
    assert mode in ("sgd", "batch")
    P = np.linspace(-2, 2, n_points).reshape(1, -1)
    T = g(P)
    params = init_params(S1, seed)
    rng = np.random.default_rng(None if seed is None else seed + 12345)
    Q = P.shape[1]
    sse = np.zeros(epochs)

    for ep in range(epochs):
        if mode == "batch":
            grads = backprop_gradients(params, P, T)
            params = [x - alpha * dx for x, dx in zip(params, grads)]
        else:
            order = rng.permutation(Q) if shuffle else np.arange(Q)
            for q in order:
                grads = backprop_gradients(params, P[:, q:q+1], T[:, q:q+1])
                params = [x - alpha * dx for x, dx in zip(params, grads)]

        _, _, a2 = forward(params, P)
        sse[ep] = np.sum((T - a2) ** 2)
    return params, sse, P, T


# ---------------------------------- experiments -------------------------------
def run_experiments(S1_list=(2, 10), alphas=(0.05, 0.2), seeds=(0, 1, 2),
                    epochs=2000, mode="batch", n_points=41):
    """
    Loops over hidden sizes, learning rates and initial conditions (seeds).
    One figure per (S1, alpha): left = network response vs g(p), right = squared error/epoch.
    """
    p_fine = np.linspace(-2, 2, 400).reshape(1, -1)
    for S1 in S1_list:
        for alpha in alphas:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            ax[0].plot(p_fine[0], g(p_fine)[0], "k--", lw=2, label="g(p)")
            for seed in seeds:
                params, sse, P, T = train(S1, alpha, epochs, mode, seed, n_points)
                _, _, out = forward(params, p_fine)
                ax[0].plot(p_fine[0], out[0], label=f"init seed {seed}")
                ax[1].semilogy(sse, label=f"init seed {seed}")
                print(f"S1={S1:3d} alpha={alpha:<6} mode={mode:5s} seed={seed} "
                      f"final SSE={sse[-1]:.5f}")
            ax[0].plot(P[0], T[0], "k.", ms=4, label="training points")
            ax[0].set(title=f"Network output (1-{S1}-1, alpha={alpha}, {mode})",
                      xlabel="p", ylabel="a")
            ax[1].set(title="Squared error per epoch", xlabel="epoch",
                      ylabel="sum of squared errors (log scale)")
            for a in ax:
                a.grid(True, alpha=0.3); a.legend(fontsize=8)
            plt.tight_layout()
    plt.show()


# ------------------------------------ main ------------------------------------
if __name__ == "__main__":
    # Edit these to experiment ------------------------------------------------
    MODE     = "batch"          # "batch" (true gradient)  or  "sgd" (stochastic)
    S1_LIST  = [2, 10]          # any number of hidden neurons, e.g. [2, 5, 10, 25]
    ALPHAS   = [0.1, 0.3]       # learning rates to compare (batch); try ~0.03-0.1 for "sgd"
    SEEDS    = [0, 1, 2]        # different initial conditions
    EPOCHS   = 50000            # batch needs many epochs (long plateau); "sgd" needs far fewer
    # -------------------------------------------------------------------------
    run_experiments(S1_LIST, ALPHAS, SEEDS, EPOCHS, MODE)
