import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ROOT       # PyROOT
from lmfit import Model, Parameters
import scipy.stats

# --- 1. load histogram data via ROOT ---
f = ROOT.TFile.Open("distros.root")
h = f.Get("dist1")
if not h:
    raise RuntimeError("Histogram 'dist1' not found in file")

h.Sumw2()  # ensure errors stored
nbins = h.GetNbinsX()
centers = []
counts  = []
errs    = []
for i in range(1, nbins+1):
    centers.append(h.GetBinCenter(i))
    counts .append(h.GetBinContent(i))
    errs   .append(h.GetBinError(i))

x_all = np.array(centers, dtype=float)
y_all = np.array(counts,  dtype=float)
err_all = np.array(errs,  dtype=float)

# Optionally mask zero‐count or zero‐error bins
mask = (y_all > 0) & (err_all > 0)
x = x_all[mask]
y = y_all[mask]
err = err_all[mask]

# --- 2. define models with lmfit ---
def two_gauss(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return ( A1 * np.exp( -0.5*((x-mu1)/sigma1)**2 )
           + A2 * np.exp( -0.5*((x-mu2)/sigma2)**2 ) )

model_gauss = Model(two_gauss, independent_vars=['x'])

def gumbel(x, N, mu, beta):
    return N * (1.0/beta) * np.exp( -(x-mu)/beta - np.exp( -(x-mu)/beta ) )

model_gumbel = Model(gumbel, independent_vars=['x'])

# --- 3. setup & run fits ---
# Sum of two Gaussians initial guesses
params_g = model_gauss.make_params(
    A1 = y.max()*0.6,
    mu1 = x[np.argmax(y)],
    sigma1 = (x.max()-x.min())*0.1,
    A2 = y.max()*0.4,
    mu2 = x[np.argmax(y)] + (x.max()-x.min())*0.2,
    sigma2 = (x.max()-x.min())*0.2
)
params_g['sigma1'].min = 0
params_g['sigma2'].min = 0

result_g = model_gauss.fit(y, params_g, x=x, weights=1.0/err)

print("=== Sum of two Gaussians fit report ===")
print(result_g.fit_report())

# Gumbel initial guesses
params_u = model_gumbel.make_params(
    N = np.sum(y)*(x[1]-x[0]),
    mu = x[np.argmax(y)]*0.5 + x.min()*0.5,
    beta = (x.max()-x.min())*0.2
)
params_u['beta'].min = 0

result_u = model_gumbel.fit(y, params_u, x=x, weights=1.0/err)

print("=== Gumbel fit report ===")
print(result_u.fit_report())

# --- 4. compute χ², dof, p-values ---
def compute_chi2(y_obs, y_pred, err, n_params):
    resid = (y_obs - y_pred) / err
    chi2 = np.sum(resid**2)
    dof  = len(y_obs) - n_params
    pval = 1.0 - scipy.stats.chi2.cdf(chi2, dof)
    return chi2, dof, pval

chi2_g, dof_g, pval_g = compute_chi2(y, result_g.best_fit, err, len(result_g.params))
chi2_u, dof_u, pval_u = compute_chi2(y, result_u.best_fit, err, len(result_u.params))

print(f"Gauss model:   χ² = {chi2_g:.1f}, dof = {dof_g}, p-value = {pval_g:.3f}")
print(f"Gumbel model:  χ² = {chi2_u:.1f}, dof = {dof_u}, p-value = {pval_u:.3f}")

# --- 5. save summary + plots to PDF ---
with PdfPages('ex1.pdf') as pdf:
    # Page 1: summary text
    plt.figure(figsize=(8,6))
    plt.axis('off')
    plt.text(0.1, 0.8, "Fit Results Summary", fontsize=14)
    plt.text(0.1, 0.70, f"Two Gaussian model: χ² = {chi2_g:.1f}, dof = {dof_g}, p = {pval_g:.3f}")
    plt.text(0.1, 0.65, f"Gumbel model:       χ² = {chi2_u:.1f}, dof = {dof_u}, p = {pval_u:.3f}")
    plt.text(0.1, 0.15, "Comments: Reduced chi square for the two gaussian is 7.82\n while tge Gumbel is 0.60, so one has the errors over estimated while the other\n is underestimated. For the p values the Gaussian method has a p=0 so it can be\n rejected while the Gumbel clearly describes it better. From these points the\n Gumbel fit is a better fit with potentially over estimated errors. ")
    pdf.savefig()
    plt.close()
    
    # Page 2: plots for both fits
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    
    axes[0].errorbar(x, y, yerr=err, fmt='o', label='data')
    axes[0].plot(x, result_g.best_fit, '-', label='two Gauss fit')
    axes[0].set_title(f"Two Gauss: p={pval_g:.3f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Counts")
    axes[0].legend()
    
    axes[1].errorbar(x, y, yerr=err, fmt='o', label='data')
    axes[1].plot(x, result_u.best_fit, '-', label='Gumbel fit')
    axes[1].set_title(f"Gumbel: p={pval_u:.3f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Counts")
    axes[1].legend()
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("Saved results and plots to ex1.pdf")

