import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================
# Parameters
# =========================
n = 6
dim = 2
epsilon = 1.0
nu = 2
beta = 0.5

# =========================
# Reference trajectory
# =========================
def r(t):
    # return np.array([t, 2 * np.sin(0.5 * t)])
    return np.array([t, 0])

def r_dot(t):
    # return np.array([1, np.cos(0.5 * t)])
    return np.array([1, 0])

# =========================
# Initial positions
# =========================
np.random.seed(42)
Z0 = 5 * np.random.randn(dim * n)

c0 = np.mean(Z0.reshape((n, dim)), axis=0)
sigma0 = c0 - r(0)
# Theoretical convergence time for centroid tracking (Proposition 1)
tau_theory = np.max(np.abs(sigma0)**(1 - beta) / (1 - beta))

# =========================
# Base Formation
# =========================
P_base = np.zeros((n, dim))
for i in range(n):
    P_base[i, :] = np.array([np.cos((i+1)*np.pi/3), np.sin((i+1)*np.pi/3)])
P_base -= np.mean(P_base, axis=0)

# =========================
# Dynamics
# =========================
def dynamics(t, Z, n, dim, M_val, epsilon, nu, beta, P):
    Zmat = Z.reshape((n, dim))
    c = np.mean(Zmat, axis=0)

    r_val = r(t)
    rd_val = r_dot(t)

    sigma = c - r_val
    zeta = np.sign(sigma) * (np.abs(sigma)**beta)

    B = Zmat - P
    diff = B[:, None, :] - B[None, :, :]
    norms = np.linalg.norm(diff, axis=2)
    denom = norms**nu + epsilon

    M_matrix = M_val * np.eye(dim)
    Md = diff @ M_matrix.T

    interaction = Md / denom[:, :, None]
    np.fill_diagonal(interaction[:, :, 0], 0)
    np.fill_diagonal(interaction[:, :, 1], 0)

    sum_term = np.sum(interaction, axis=1) / n
    dZ = -Zmat - sum_term + r_val + rd_val - zeta + P
    return dZ.flatten()

def compute_rho(M_val, nu, epsilon):
    # Theoretical formation error bound (Proposition 2, Eq 11)
    if nu > 1:
        return (M_val / nu) * ((nu - 1) / epsilon)**((nu - 1) / nu)
    else:
        return M_val * (n - 1) / n  # Fallback for nu=1

def extract_metrics(sol, P):
    # sol.y shape: (n*dim, timesteps)
    Z_traj = sol.y.T  # (timesteps, n*dim)
    t_vals = sol.t
    
    sigma_norms = np.zeros(len(t_vals))
    max_delta_norms = np.zeros(len(t_vals))
    
    for k, t in enumerate(t_vals):
        Zmat = Z_traj[k].reshape((n, dim))
        c = np.mean(Zmat, axis=0)
        r_val = r(t)
        
        # Centroid error
        sigma_norms[k] = np.linalg.norm(c - r_val)
        
        # Formation error for each agent: delta_i = x_i - p_i - r
        deltas = Zmat - P - r_val
        delta_norms = np.linalg.norm(deltas, axis=1)
        max_delta_norms[k] = np.max(delta_norms)
        
    return sigma_norms, max_delta_norms

# =========================
# Simulations
# =========================
t_span = (0, 30)
t_eval = np.linspace(0, 30, 500)

M_values = [1.0, 5.0, 10.0]
L_fixed = 5.0
results_M = {}

print("Running Experiment 1: Varying M...")
for M_val in M_values:
    P = L_fixed * P_base
    sol = solve_ivp(dynamics, t_span, Z0, args=(n, dim, M_val, epsilon, nu, beta, P), t_eval=t_eval)
    sigma_norms, max_delta_norms = extract_metrics(sol, P)
    rho = compute_rho(M_val, nu, epsilon)
    
    delta0 = Z0.reshape((n, dim)) - P - r(0)
    delta0_norms = np.linalg.norm(delta0, axis=1)
    t_bars = np.array([- (n / 2) * np.log(rho**2 / d**2) if d > rho else 0 for d in delta0_norms])
    t_form_theory = max(tau_theory, np.max(t_bars))
    
    prac_form_idx = np.where(max_delta_norms > rho)[0]
    t_form_prac = sol.t[prac_form_idx[-1] + 1] if len(prac_form_idx) > 0 and prac_form_idx[-1] + 1 < len(sol.t) else 0

    results_M[M_val] = {
        't': sol.t,
        'sigma': sigma_norms,
        'delta': max_delta_norms,
        'rho': rho,
        't_form_theory': t_form_theory,
        't_form_prac': t_form_prac
    }

L_values = [2.0, 5.0, 10.0]
M_fixed = 5.0
results_L = {}

print("Running Experiment 2: Varying L...")
for L_val in L_values:
    P = L_val * P_base
    sol = solve_ivp(dynamics, t_span, Z0, args=(n, dim, M_fixed, epsilon, nu, beta, P), t_eval=t_eval)
    sigma_norms, max_delta_norms = extract_metrics(sol, P)
    rho = compute_rho(M_fixed, nu, epsilon)
    
    delta0 = Z0.reshape((n, dim)) - P - r(0)
    delta0_norms = np.linalg.norm(delta0, axis=1)
    t_bars = np.array([- (n / 2) * np.log(rho**2 / d**2) if d > rho else 0 for d in delta0_norms])
    t_form_theory = max(tau_theory, np.max(t_bars))
    
    prac_form_idx = np.where(max_delta_norms > rho)[0]
    t_form_prac = sol.t[prac_form_idx[-1] + 1] if len(prac_form_idx) > 0 and prac_form_idx[-1] + 1 < len(sol.t) else 0

    results_L[L_val] = {
        't': sol.t,
        'sigma': sigma_norms,
        'delta': max_delta_norms,
        'rho': rho,
        't_form_theory': t_form_theory,
        't_form_prac': t_form_prac
    }

print("Simulations complete. Generating plots...")

# =========================
# Plotting
# =========================

def moving_average(x, w=10):
    return np.convolve(x, np.ones(w), 'valid') / w
window = 10

plt.figure(figsize=(14, 10))

# Plot 1: Centroid Error (Varying M)
plt.subplot(2, 2, 1)
for M_val in M_values:
    plt.plot(results_M[M_val]['t'], results_M[M_val]['sigma'], label=f'M={M_val}')
plt.axvline(tau_theory, color='k', linestyle='--', label=f'Theoretical τ={tau_theory:.2f}s')
plt.title(f'Centroid Tracking Error |σ(t)| (Varying M, fixed L={L_fixed})')
plt.xlabel('Time (s)')
plt.ylabel('|σ(t)|')
plt.xlim(0, 5) # Updated limit
plt.legend()
plt.grid(True)

# Plot 2: Formation Error (Varying M)
plt.subplot(2, 2, 2)
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, M_val in enumerate(M_values):
    res = results_M[M_val]
    delta_ma = moving_average(res['delta'], window)
    t_ma = res['t'][window-1:]
    plt.plot(t_ma, delta_ma, color=colors[i], label=f'M={M_val}')
    plt.axhline(res['rho'], color=colors[i], linestyle='--', label=f'Bound ρ={res["rho"]:.1f}')
    # plt.axvline(res['t_form_theory'], color=colors[i], linestyle=':', label=f'Thry τ_f={res["t_form_theory"]:.2f}s')
    # plt.axvline(res['t_form_prac'], color=colors[i], linestyle='-.', label=f'Prac τ_f={res["t_form_prac"]:.2f}s')
plt.title(f'Max Formation Error max_i|δ_i(t)| (MA) (Varying M, fixed L={L_fixed})')
plt.xlabel('Time (s)')
plt.ylabel('max_i|δ_i(t)|')
# plt.ylim(0, 2e-5)
plt.xlim(0, 15)
plt.legend()
plt.grid(True)

# Plot 3: Centroid Error (Varying L)
plt.subplot(2, 2, 3)
for L_val in L_values:
    plt.plot(results_L[L_val]['t'], results_L[L_val]['sigma'], label=f'L={L_val}')
plt.axvline(tau_theory, color='k', linestyle='--', label=f'Theoretical τ={tau_theory:.2f}s')
plt.title(f'Centroid Tracking Error |σ(t)| (Varying L, fixed M={M_fixed})')
plt.xlabel('Time (s)')
plt.ylabel('|σ(t)|')
plt.xlim(0, 5)
plt.legend()
plt.grid(True)

# Plot 4: Formation Error (Varying L)
plt.subplot(2, 2, 4)
for i, L_val in enumerate(L_values):
    res = results_L[L_val]
    delta_ma = moving_average(res['delta'], window)
    t_ma = res['t'][window-1:]
    plt.plot(t_ma, delta_ma, color=colors[i], label=f'L={L_val}')
    # plt.axvline(res['t_form_theory'], color=colors[i], linestyle=':', label=f'Thry τ_f={res["t_form_theory"]:.2f}s')
    # plt.axvline(res['t_form_prac'], color=colors[i], linestyle='-.', label=f'Prac τ_f={res["t_form_prac"]:.2f}s')

rho_fixed = compute_rho(M_fixed, nu, epsilon)
plt.axhline(rho_fixed, color='k', linestyle='--', label=f'Bound ρ={rho_fixed:.1f}')

# Add representative theory/practical lines for the last L_val to avoid clutter
res_rep = results_L[L_values[-1]]
# plt.axvline(res_rep['t_form_theory'], color='r', linestyle=':', label=f'Thry τ_f (all L)={res_rep["t_form_theory"]:.2f}s')
# plt.axvline(res_rep['t_form_prac'], color='r', linestyle='-.', label=f'Prac τ_f (L={L_values[-1]})={res_rep["t_form_prac"]:.2f}s')

plt.title(f'Max Formation Error max_i|δ_i(t)| (MA) (Varying L, fixed M={M_fixed})')
plt.xlabel('Time (s)')
plt.ylabel('max_i|δ_i(t)|')
# plt.ylim(0, 2e-5)
plt.xlim(0, 15)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('theory_validation_plots.png', dpi=500)
print("Saved plot to 'theory_validation_plots.png'")
# plt.show() # Disabled so it doesn't block the script
