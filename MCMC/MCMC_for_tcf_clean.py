
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import h5py
from matplotlib.colors import LogNorm

import emcee
import corner
import joblib


###############################################################################################################################################
###############################################################################################################################################
# # 1. Open Data
###############################################################################################################################################
###############################################################################################################################################

###############################################################################################################################################
# ## 1. a. Covariance Matrix
###############################################################################################################################################


cov_matrix = np.load("/home/lcrascal/Code/TCF/TCF_completed_code/tcf_gpu_loreli_mcmc/MCMC/Full_Cov_Matrix.npy")

# # cut cov matrix to only use first half
# half = int((np.shape(cov_matrix)[0]+1)/2)
# cov_matrix = cov_matrix[0:half, 0:half]

plt.imshow(cov_matrix)
plt.colorbar()
plt.title("Covariance Matrix")
plt.show()
plt.close()

plt.imshow(np.linalg.inv(cov_matrix))
plt.colorbar()
plt.title("Inverse Covariance Matrix")
plt.show()
plt.close()


###############################################################################################################################################
# ## 1. b. Observed Data
###############################################################################################################################################



# all TCF data
all_TCF_data_df = pd.read_pickle("/home/lcrascal/Code/TCF/TCF_completed_code/tcf_gpu_loreli_mcmc/Emulator/LoReLi_tcf_dataframe.pkl")

# choose given observed sim
sim_num = 11364 #14364
#sim_num = 11534

mask = all_TCF_data_df["sim_name"].astype(str) == f"{sim_num}"
obs_data_mean = all_TCF_data_df.loc[mask, "sr_mean"].iloc[0]
obs_data_std = all_TCF_data_df.loc[mask, "sr_std"].iloc[0] # do I need this data at all?

# halving the data
obs_data_mean = obs_data_mean[0:half]
obs_data_std = obs_data_std[0:half]

# params of obs TCF
params_df = pd.read_pickle(f"/data/cluster/emc-brid/Datasets/LoReLi/metadata/LoReLi_database_loggedparams.pkl")
params_df = params_df.sort_index()

obs_params = (params_df.loc[f"{sim_num}"]).tolist()
print(obs_params)

param_names = ["fX", "rHS", "tau", "Mmin", "fesc"]

#display(params_df)

###############################################################################################################################################
# ## 1. c. The Saved Emulator
###############################################################################################################################################



emu = joblib.load("/home/lcrascal/Code/TCF/TCF_completed_code/tcf_gpu_loreli_mcmc/Emulator/MPL_test_tcf_emulator.joblib")

mlp_model = emu["model"]
param_names = emu["param_names"]
r = emu["r"]


def emulator_predict(params, model):
    """
    predict TCF data with emulator 
    
    1. provide given params
    2. predict TCF at those params
    3. return TCF data
    
    """
    params = np.asarray(params, dtype=float).reshape(1, -1)  # (1, ndim)

    y_pred = model.predict(params)  # (1, n_r)
    
    y_pred = y_pred.ravel()#[0:25] # this is the case for when I have rediced the size of the covariance matrix
    
    return y_pred

###############################################################################################################################################
###############################################################################################################################################
# # 2. The Functions
###############################################################################################################################################
###############################################################################################################################################


###############################################################################################################################################
# ## 2. a. Prior Function
###############################################################################################################################################


def log_prior(log_params):

    """
    With uniform priors for fX, rHS, Mmin and fesc
    But with a coupled non-unifrom prior for tau-Mmin
    """

    
    logfX, rHS, tau, logMmin, fesc = log_params

    # 1D bounds
    if not (-3.71669877129645033 < logfX < -1.7166987712964503):
        return -np.inf
    if not (0.0 < rHS < 1.0):
        return -np.inf
    if not (2.8685856665587655 < tau < 4.021383654057061):
        return -np.inf
    if not (8.0 < logMmin < 9.6):
        return -np.inf
    if not (0.05 < fesc < 0.5):
        return -np.inf

    # 2D coupled prior
    tau_mmin_val = tau + 0.38 * logMmin
    if not (6.52 < tau_mmin_val < 7.07):
        return -np.inf

    return 0.0


###############################################################################################################################################
# ## 2. b. Likelihood Function
###############################################################################################################################################


def log_likelihood(log_params, obs_data, cov, model_func):
    """
    Multivariate Gaussian log-likelihood:
        ln L = residuial^T * C^{-1} * residual 
    where residuial = obs_data - model(params).

    """
    obs_data = np.asarray(obs_data, dtype=float)
    model_data = np.asarray(model_func(log_params), dtype=float)
    cov_matrix = np.asarray(cov, dtype=float)
    
    residuals = obs_data - model_data

    logL = residuals @ np.linalg.solve(cov_matrix, residuals)
    print("logL", logL)

    return -0.5*logL


###############################################################################################################################################
# ## 2. c. Posterior Function
###############################################################################################################################################


def log_posterior(params, obs_data, cov, model_func):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(params, obs_data, cov, model_func)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll



###############################################################################################################################################
###############################################################################################################################################
# # 3. The MCMC
###############################################################################################################################################
###############################################################################################################################################


###############################################################################################################################################
# ## 3. a. Set Up MCMC
###############################################################################################################################################


ndim = 5
nwalkers = 32

# choosing walker starting points randomly from within priors:
def choose_sample_from_prior(n_samples):
    samples = []

    while len(samples) < n_samples:
        # draw from 1D bounds
        logfX   = np.random.uniform(-3.71669877129645033, -1.7166987712964503)
        rHS     = np.random.uniform(0.0, 1.0)
        tau     = np.random.uniform(2.8685856665587655, 4.021383654057061)
        logMmin = np.random.uniform(8.0, 9.6)
        fesc    = np.random.uniform(0.05, 0.5)

        trial = np.array([logfX, rHS, tau, logMmin, fesc])

        # keep only if inside full prior
        if np.isfinite(log_prior(trial)):
            samples.append(trial)

    return np.array(samples)

p0 = choose_sample_from_prior(nwalkers)

print("p0 shape:", p0.shape)
print(p0[:5])


MCMC_sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_posterior,
    args=[obs_data_mean, cov_matrix, lambda p: emulator_predict(p, mlp_model)]
)


###############################################################################################################################################
# ## 3. b. Run The MCMC
###############################################################################################################################################



# burn in 
MCMC_sampler.reset()
state = MCMC_sampler.run_mcmc(p0, 100, progress=True)
MCMC_sampler.reset() 

# full run
MCMC_sampler.run_mcmc(state, 2000, progress=True);


###############################################################################################################################################
###############################################################################################################################################
# # 4. Results
###############################################################################################################################################
###############################################################################################################################################


run_tag = f"XXX"
outdir = Path(f"./mcmc_outputs/{run_tag}")
outdir.mkdir(parents=True, exist_ok=True)



###############################################################################################################################################
# ## 4. a. Found Values of Parameters
###############################################################################################################################################

chain = MCMC_sampler.get_chain()                 # (nsteps, nwalkers, ndim)
logp  = MCMC_sampler.get_log_prob()              # (nsteps, nwalkers)
acc_frac = MCMC_sampler.acceptance_fraction      # (nwalkers,)

samples = chain.reshape(-1, chain.shape[-1])     # flattened samples

# Save compressed raw outputs
np.savez_compressed(
    outdir / "chain_and_logp.npz",
    chain=chain.astype(np.float32),
    logp=logp.astype(np.float32),
    acceptance_fraction=acc_frac.astype(np.float32),
    obs_data=np.asarray(obs_data, dtype=np.float32),
    cov_matrix=np.asarray(cov_matrix, dtype=np.float32),
    obs_params=np.asarray(obs_params, dtype=np.float32),
    param_labels=np.array(param_labels, dtype=object),
    obs_sim_name=f"{sim_num}",
    zgroup="zsim_7p431",
)

# Save flat samples separately too
np.save(outdir / "flat_samples.npy", samples.astype(np.float32))

def posterior_summary(samples, labels, truths=None):
    q16, q50, q84 = np.quantile(samples, [0.16, 0.5, 0.84], axis=0)
    mean = np.mean(samples, axis=0)
    std  = np.std(samples, axis=0)

    df = pd.DataFrame({
        "param": labels,
        "mean": mean,
        "std": std,
        "q16": q16,
        "median": q50,
        "q84": q84,
        "err_minus": q50 - q16,
        "err_plus": q84 - q50,
    })

    if truths is not None:
        df["truth"] = truths

    return df

summary_df = posterior_summary(samples, param_labels, truths=np.asarray(obs_params))
summary_df.to_csv(outdir / "posterior_summary_full.csv", index=False)

with open(outdir / "run_info.txt", "w") as f:
    f.write(f"chain shape: {chain.shape}\n")
    f.write(f"flat samples shape: {samples.shape}\n")
    f.write(f"mean acceptance fraction: {np.mean(acc_frac):.5f}\n")
    f.write(f"min acceptance fraction: {np.min(acc_frac):.5f}\n")
    f.write(f"max acceptance fraction: {np.max(acc_frac):.5f}\n")
    

###############################################################################################################################################
# ## 4. b. Corner Plot
###############################################################################################################################################


ranges = [
    (-3.9, -1.5),   # logfX
    (-0.105, 1.105),     # rHS
    (2.8, 4.1),      # tau
    (7.9, 9.7),      # logMmin
    (0.0, 0.55),    # fesc
]


fig = corner.corner(
    samples,
    labels=param_names,
    truths=obs_params,
    range=ranges,
    show_titles=True,
    title_fmt=".3f",
    #quantiles=[0.16, 0.5, 0.84],
    levels=(0.68, 0.95),
    plot_density=False,
    plot_contours=True,
    fill_contours=True,
    plot_datapoints=False,
    #smooth=1.0,          # try 0.5–2.0
    bins=50,
    color="crimson"
)


# adding the prior contors

ndim = len(param_names)
axes = np.array(fig.axes).reshape((ndim, ndim))

for i in range(ndim):
    for j in range(ndim):
        ax = axes[i, j]

        # corner only uses lower triangle
        if i < j:
            continue

        xparam = param_names[j]
        yparam = param_names[i]

        xmin, xmax = bounds[xparam]
        ymin, ymax = bounds[yparam]

        # Diagonal panels: draw 1D top-hat bounds
        if i == j:
            ax.axvline(xmin, color="black", lw=1.5)
            ax.axvline(xmax, color="black", lw=1.5)
            continue

        # Draw rectangle for all 2D panels
        ax.plot([xmin, xmax], [ymin, ymin], color="black", lw=1.2)
        ax.plot([xmin, xmax], [ymax, ymax], color="black", lw=1.2)
        ax.plot([xmin, xmin], [ymin, ymax], color="black", lw=1.2)
        ax.plot([xmax, xmax], [ymin, ymax], color="black", lw=1.2)

        # Special case: tau vs logMmin panel
        if xparam == "tau" and yparam == "Mmin":
            tau_grid = np.linspace(bounds["tau"][0], bounds["tau"][1], 300)

            mmin_line1 = (6.52 - tau_grid) / 0.38
            mmin_line2 = (7.07 - tau_grid) / 0.38

            # Only draw the parts inside the box
            mask1 = (mmin_line1 >= bounds["Mmin"][0]) & (mmin_line1 <= bounds["Mmin"][1])
            mask2 = (mmin_line2 >= bounds["Mmin"][0]) & (mmin_line2 <= bounds["Mmin"][1])

            ax.plot(tau_grid[mask1], mmin_line1[mask1], color="black", lw=1.5)
            ax.plot(tau_grid[mask2], mmin_line2[mask2], color="black", lw=1.5)

        # Also handle the transposed panel if needed
        elif xparam == "logMmin" and yparam == "tau":
            mmin_grid = np.linspace(bounds["Mmin"][0], bounds["Mmin"][1], 300)

            tau_line1 = 6.52 - 0.38 * mmin_grid
            tau_line2 = 7.07 - 0.38 * mmin_grid

            mask1 = (tau_line1 >= bounds["tau"][0]) & (tau_line1 <= bounds["tau"][1])
            mask2 = (tau_line2 >= bounds["tau"][0]) & (tau_line2 <= bounds["tau"][1])

            ax.plot(mmin_grid[mask1], tau_line1[mask1], color="black", lw=1.5)
            ax.plot(mmin_grid[mask2], tau_line2[mask2], color="black", lw=1.5)

fig.savefig(f"{outdir}/RESULT_corner_plot.png")
plt.close(fig)


###############################################################################################################################################
# ## 4. c. Plot 2D Walker Paths
###############################################################################################################################################


def plot_walkers_2d(chain, i, j, labels=None, truths=None,
                    traj_stride=5, n_traj_show=6, seed=0):
    """
    chain: (nsteps, nwalkers, ndim)
    i, j: parameter indices for x and y axes
    """
    rng = np.random.default_rng(seed)
    nsteps, nwalkers, ndim = chain.shape

    x = chain[:, :, i]
    y = chain[:, :, j]

    # Choose a small subset of walkers to show trajectories
    idx = np.arange(nwalkers)
    if n_traj_show < nwalkers:
        idx = rng.choice(idx, size=n_traj_show, replace=False)

    plt.figure(figsize=(7, 5))

    # Faint trajectories for a few walkers
    for w in idx:
        plt.plot(x[::traj_stride, w], y[::traj_stride, w],
                 color="gray", alpha=0.35, linewidth=0.8)

    # Ensemble mean path (all walkers) through time
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)
    plt.plot(mean_x[::traj_stride], mean_y[::traj_stride],
             color="crimson", linewidth=2, label="Ensemble mean path")

    # Final walker positions (helps show convergence region)
    plt.scatter(x[-1, :], y[-1, :], s=12, color="black", alpha=0.6, label="Final walker positions")

    # Truth marker if available
    if truths is not None:
        plt.plot(truths[i], truths[j], marker="x", markersize=10, color="black", label="Truth")

    xl = labels[i] if labels is not None else f"param {i}"
    yl = labels[j] if labels is not None else f"param {j}"
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(f"Walker motion in parameter space: {xl} vs {yl}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/2D_walkers_params_{i}_{j}.png")
    plt.close()

plot_walkers_2d(chain, 0, 1, labels=param_names, truths=obs_params, traj_stride=1, n_traj_show=32)
plot_walkers_2d(chain, 0, 2, labels=param_names, truths=obs_params, traj_stride=1, n_traj_show=32)
plot_walkers_2d(chain, 0, 3, labels=param_names, truths=obs_params, traj_stride=1, n_traj_show=32)
plot_walkers_2d(chain, 0, 4, labels=param_names, truths=obs_params, traj_stride=1, n_traj_show=32)
plot_walkers_2d(chain, 1, 2, labels=param_names, truths=obs_params, traj_stride=1, n_traj_show=32)
plot_walkers_2d(chain, 1, 3, labels=param_names, truths=obs_params, traj_stride=1, n_traj_show=32)
plot_walkers_2d(chain, 1, 4, labels=param_names, truths=obs_params, traj_stride=1, n_traj_show=32)
plot_walkers_2d(chain, 2, 3, labels=param_names, truths=obs_params, traj_stride=1, n_traj_show=32)
plot_walkers_2d(chain, 2, 4, labels=param_names, truths=obs_params, traj_stride=1, n_traj_show=32)
plot_walkers_2d(chain, 3, 4, labels=param_names, truths=obs_params, traj_stride=1, n_traj_show=32)


def plot_trace(chain_param, true_value, label):
    """
    chain_param shape: (nsteps, nwalkers)
    """
    nsteps, nwalkers = chain_param.shape
    
    plt.figure(figsize=(8, 4))
    
    # Plot walkers faintly
    for w in range(nwalkers):
        plt.plot(chain_param[:, w], color="gray", alpha=0.3, linewidth=0.5)
    
    # Plot ensemble mean (strong line)
    mean_trace = chain_param.mean(axis=1)
    plt.plot(mean_trace, color="crimson", linewidth=2, label="Ensemble mean")
    
    if true_value is not None:
        plt.axhline(true_value, color="black", linestyle="--", label="True value")
    
    plt.xlabel("Step")
    plt.ylabel(label)
    plt.title(f"Trace plot: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/1D_walkers_param_{label}.png")
    plt.close()


plot_trace(chain[:, :, 0], true_value=obs_params[0], label="fX")
plot_trace(chain[:, :, 1], true_value=obs_params[1], label="rHS")
plot_trace(chain[:, :, 2], true_value=obs_params[2], label="tau")
plot_trace(chain[:, :, 3], true_value=obs_params[3], label="Mmin")
plot_trace(chain[:, :, 4], true_value=obs_params[4], label="fesc")





