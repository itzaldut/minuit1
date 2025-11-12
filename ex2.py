import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ROOT
from lmfit import Parameters, Minimizer, fit_report
import scipy.stats

f = ROOT.TFile.Open("experiments.root")
h1 = f.Get("hexp1")
h2 = f.Get("hexp2")
if not h1 or not h2:
    raise RuntimeError("Histograms hexp1 or hexp2 not found")

h1.Sumw2()
h2.Sumw2()

x1, y1, err1 = [], [], []
for i in range(1, h1.GetNbinsX()+1):
    x1.append(h1.GetBinCenter(i))
    y1.append(h1.GetBinContent(i))
    err1.append(h1.GetBinError(i))
x2, y2, err2 = [], [], []
for i in range(1, h2.GetNbinsX()+1):
    x2.append(h2.GetBinCenter(i))
    y2.append(h2.GetBinContent(i))
    err2.append(h2.GetBinError(i))

x1 = np.array(x1, dtype=float)
y1 = np.array(y1, dtype=float)
err1 = np.array(err1, dtype=float)
x2 = np.array(x2, dtype=float)
y2 = np.array(y2, dtype=float)
err2 = np.array(err2, dtype=float)

mask1 = (y1>0) & (err1>0)
mask2 = (y2>0) & (err2>0)
x1, y1, err1 = x1[mask1], y1[mask1], err1[mask1]
x2, y2, err2 = x2[mask2], y2[mask2], err2[mask2]


def signal(x, mean, sigma, amp):
    return amp * (1.0/(sigma * np.sqrt(2*np.pi))) * np.exp( -0.5*((x-mean)/sigma)**2 )

def background1(x, lam, bkg_amp1):
    return bkg_amp1 * np.exp(-x/lam)

def background2(x, n, bkg_amp2):
    return bkg_amp2 * x**n

# Objective: combined residuals
def objective(params, x1, y1, err1, x2, y2, err2):
    mean  = params['mean']
    sigma = params['sigma']
    amp1  = params['amp1']
    lam   = params['lam']
    bkg1_amp = params['bkg1_amp']
    amp2  = params['amp2']
    n     = params['n']
    bkg2_amp = params['bkg2_amp']

    model1 = signal(x1, mean, sigma, amp1) + background1(x1, lam, bkg1_amp)
    model2 = signal(x2, mean, sigma, amp2) + background2(x2, n, bkg2_amp)

    resid1 = (y1 - model1)/err1
    resid2 = (y2 - model2)/err2
    return np.concatenate([resid1, resid2])

#Parameters
params = Parameters()
params.add('mean',    value=75.0,   min=0)
params.add('sigma',   value=5.0,    min=0)
params.add('amp1',    value=2000.0,  min=0)
params.add('lam',     value=20.0,   min=0)
params.add('bkg1_amp',value=100.0,  min=0)
params.add('amp2',    value=2000.0,  min=0)
params.add('n',       value=-2.0)
params.add('bkg2_amp',value=100.0,  min=0)

#Fit
minimizer = Minimizer(objective, params,
                      fcn_args=(x1, y1, err1, x2, y2, err2))
result = minimizer.minimize(method='leastsq')

print("=== Simultaneous Fit Report ===")
print(fit_report(result))

# Extract best-fit values
mean_val   = result.params['mean'].value
mean_err   = result.params['mean'].stderr
sigma_val  = result.params['sigma'].value
sigma_err  = result.params['sigma'].stderr

chi2 = result.chisqr
dof  = result.nfree
pval = 1.0 - scipy.stats.chi2.cdf(chi2, dof)
if mean_err is None:
    err_str = "±N/A"
else:
    err_str = f"±{mean_err:.3f}"

print(f"Signal mean   = {mean_val:.3f}")
print(f"Signal sigma  = {sigma_val:.3f}")
print(f"Combined χ²   = {chi2:.1f}, dof = {dof}, p-value = {pval:.3f}")

# Model curves for plotting
model1 = signal(x1, mean_val, sigma_val, result.params['amp1'].value) + \
         background1(x1, result.params['lam'].value, result.params['bkg1_amp'].value)
model2 = signal(x2, mean_val, sigma_val, result.params['amp2'].value) + \
         background2(x2, result.params['n'].value, result.params['bkg2_amp'].value)

#PDF
with PdfPages('ex2.pdf') as pdf:
    # Page 1: summary
    plt.figure(figsize=(8,6))
    plt.axis('off')
    plt.text(0.1, 0.78, "Simultaneous Fit Results", fontsize=14)
    plt.text(0.1, 0.70, f"Signal mean   = {mean_val:.3f}")
    plt.text(0.1, 0.65, f"Signal sigma  = {sigma_val:.3f}")
    plt.text(0.1, 0.60, f"Combined χ²    = {chi2:.1f}, dof = {dof}, p-value = {pval:.3f}")
    plt.text(0.1, 0.30, "Comments:\n I think this is not a good fit, a p value of\n 0 suggests this is not a likely good fit since it is much under 0.05. The chi\n square value is also larger than 1. ",
             fontsize=8, multialignment='left')
    pdf.savefig()
    plt.close()

    # Page 2: plot both distributions with fit overlays
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    axes[0].errorbar(x1, y1, yerr=err1, fmt='o', label='Data Exp1')
    axes[0].plot(x1, model1, '-', label='Fit Exp1')
    axes[0].set_title("Experiment 1")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Counts")
    axes[0].legend()

    axes[1].errorbar(x2, y2, yerr=err2, fmt='o', label='Data Exp2')
    axes[1].plot(x2, model2, '-', label='Fit Exp2')
    axes[1].set_title("Experiment 2")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Counts")
    axes[1].legend()

    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("Saved results and plot in ex2.pdf")

