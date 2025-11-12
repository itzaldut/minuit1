import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ROOT
from lmfit import Parameters, Minimizer, fit_report
import scipy.stats

f = ROOT.TFile.Open("fitInputs.root", "READ")
hdata = f.Get("hdata")
hbkg  = f.Get("hbkg")
if (not hdata) or (not hbkg):
    raise RuntimeError("Could not find histograms hdata or hbkg")

hdata.Sumw2()
hbkg.Sumw2()

# Extract bin info
nx, ny = hdata.GetNbinsX(), hdata.GetNbinsY()
x_edges = [hdata.GetXaxis().GetBinLowEdge(i) for i in range(1, nx+2)]
y_edges = [hdata.GetYaxis().GetBinLowEdge(j) for j in range(1, ny+2)]
x_centers = np.array([(x_edges[i]+x_edges[i+1])/2 for i in range(nx)])
y_centers = np.array([(y_edges[j]+y_edges[j+1])/2 for j in range(ny)])

# Fill numpy arrays
X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
data2d = np.array([[hdata.GetBinContent(i+1,j+1) for j in range(ny)] for i in range(nx)])
bkg_template = np.array([[hbkg.GetBinContent(i+1,j+1) for j in range(ny)] for i in range(nx)])

x_flat = X.ravel()
y_flat = Y.ravel()
data_flat = data2d.ravel()
err_flat = np.sqrt(np.maximum(data_flat, 1))

mask = np.isfinite(data_flat) & (err_flat > 0)
x_flat, y_flat, data_flat, err_flat = x_flat[mask], y_flat[mask], data_flat[mask], err_flat[mask]
bkg_flat = bkg_template.ravel()[mask]

def signal2d(x, y, p0, p1, p2, p3, p4):
    return p0 * np.exp(-0.5*((x-p1)/max(p2,1e-6))**2 - 0.5*((y-p3)/max(p4,1e-6))**2)

def bkg_model_flat(bn):
    return np.clip(bn * bkg_flat,0, 1e9)


def residuals(params):
    p0 = params['p0']
    p1 = params['p1']
    p2 = params['p2']
    p3 = params['p3']
    p4 = params['p4']
    bn = params['bn']
    sig = signal2d(x_flat, y_flat, p0, p1, p2, p3, p4)
    bkg = bkg_model_flat(bn)
    model = sig + bkg
    resid = (data_flat - model) / err_flat
    return np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)

#Parameters
params = Parameters()
params.add('p0', value=500,  min=0, vary=True)    # amplitude
params.add('p1', value=np.mean(x_centers), min=min(x_centers), max=max(x_centers), vary=True)
params.add('p2', value=np.std(x_centers)/2, min=0.1, vary=True)
params.add('p3', value=np.mean(y_centers), min=min(y_centers), max=max(y_centers), vary=True)
params.add('p4', value=np.std(y_centers)/2, min=0.1, vary=True)
params.add('bn', value=1.0,  min=0, vary=True)    # background normalization


minimizer = Minimizer(residuals, params, nan_policy='omit')
result = minimizer.minimize(method='leastsq')

print("=== Fit Report ===")
print(fit_report(result))

# Check covariance
if result.covar is None:
    print("⚠️ Covariance matrix not estimated — uncertainties may be unreliable.")

# Check residuals
resid_check = residuals(result.params)
if np.allclose(resid_check, 0):
    print("⚠️ Residuals are all zero — fit may have failed.")
elif np.any(np.isnan(resid_check)):
    print("⚠️ Residuals contain NaNs — check model or data.")
else:
    print("Residuals look valid.")

def safe_val(param):
    return param.value, param.stderr if param.stderr is not None else 0.0

p0_val, p0_err = safe_val(result.params['p0'])
p1_val, p1_err = safe_val(result.params['p1'])
p2_val, p2_err = safe_val(result.params['p2'])
p3_val, p3_err = safe_val(result.params['p3'])
p4_val, p4_err = safe_val(result.params['p4'])
bn_val, bn_err = safe_val(result.params['bn'])

# Chi² and p-value
chi2 = result.chisqr
dof = result.nfree
pvalue = 1.0 - scipy.stats.chi2.cdf(chi2, dof)
print(f"χ² = {chi2:.1f}, dof = {dof}, p-value = {pvalue:.3f}")

# signal event estimation
signal_flat = signal2d(x_flat, y_flat, p0_val, p1_val, p2_val, p3_val, p4_val)
total_signal = np.sum(signal_flat)
err_signal = (p0_err/p0_val)*total_signal if p0_val>0 else 0.0
print(f"Total signal events ≈ {total_signal:.1f} ± {err_signal:.1f}")

#PDF
with PdfPages("ex3.pdf") as pdf:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={"projection": "3d"})

    # Data
    ax = axes[0, 0]
    ax.plot_surface(X, Y, data2d, cmap="viridis")
    ax.set_title("Data Histogram")

    # Fit result
    model2d = signal_flat.reshape(nx, ny) + bn_val * bkg_template
    ax = axes[0, 1]
    ax.plot_surface(X, Y, model2d, cmap="plasma")
    ax.set_title("Fit Result")

    # Residuals
    resid2d = data2d - model2d
    ax = axes[1, 0]
    ax.plot_surface(X, Y, resid2d, cmap="coolwarm")
    ax.set_title("Residuals")

    # Data minus background
    dat_minus_bkg = data2d - (bn_val * bkg_template)
    ax = axes[1, 1]
    ax.plot_surface(X, Y, dat_minus_bkg, cmap="cividis")
    ax.set_title("Data - Background")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Summary Page
    fig_summary = plt.figure(figsize=(8,6))
    plt.axis("off")
    y0, dy = 0.85, 0.05
    plt.text(0.1, y0, "2D Fit Results Summary", fontsize=14); y0 -= dy
    for label, val, err in [("p0 (amp)",p0_val,p0_err),("p1 (mean x)",p1_val,p1_err),
                            ("p2 (sigma x)",p2_val,p2_err),("p3 (mean y)",p3_val,p3_err),
                            ("p4 (sigma y)",p4_val,p4_err),("bn (bkg norm)",bn_val,bn_err)]:
        plt.text(0.1, y0, f"{label:<14}= {val:.3f} ± {err:.3f}"); y0 -= dy
    plt.text(0.1, y0, f"χ² = {chi2:.1f}, dof = {dof}, p-value = {pvalue:.3f}"); y0 -= dy
    plt.text(0.1, y0, f"Total signal events ≈ {total_signal:.1f} ± {err_signal:.1f}")
    y0 -= 2*dy
    plt.text(0.1, 0.25, "The signal events and error are given above, the number of\n signal events is obtained from summing over the bins. The error is obtained using\n the parameter p0 where we take p0 to be the value and sigmap0 to be the error,\n then the error on number of events is number of evets x (sigmap0/po)")
    pdf.savefig(fig_summary)
    plt.close(fig_summary)

print("Saved as ex3.pdf")

