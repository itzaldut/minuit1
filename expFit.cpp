// A simple C++ program to illustrate the use of ROOT class TMinuit 
// for function minimization.  The example shows a Maximum Likelihood
// fit for the mean of an exponential pdf in which TMinuit 
// minimizes - 2*log(L).   The user must specify what to minimize in the 
// function fcn, shown in this file below.

// fcn passes back f = -2*ln L by reference; this is the function to minimize.
// The factor of -2 allows MINUIT to get the errors using the same
// recipe as for least squares, i.e., go up from the minimum by 1.

// TMinuit does not allow fcn to be a member function, and the function
// arguments are fixed, so one of the only ways to bring the data  
// into fcn is to declare a pointer to the data (xVecPtr) as global.

// For more info on TMinuit see root.cern.ch/root/html/TMinuit.html .

// Based on example by 
// Glen Cowan
// RHUL Physics
// 4 December 2006

// Update Bob Hirosky: Sep 2013


#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

#include <TMinuit.h>
#include <TRandom2.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TMath.h>
#include <TH1F.h>
#include <TF1.h>
#include <TString.h>
#include <TAxis.h>
#include <TLine.h>

using namespace std;

// ---------- GLOBALS ----------
// Declare pointer to data as global
// The use of global variables is a disfavored coding practice, but
// in this case it will allow us to avoid writing more complex code
TH1F *hdata;
// also passing the fit function, to make the objective fcn more generic
TF1 *fparam;

//-------------------------------------------------------------------------
// The pdf to be fitted, here an exponential.
// This is a convenient interface for associating the model
// with a TF1 later
double expPdf(double* xPtr, double par[]){        
  double x = *xPtr;
  double A = par[0];         // normalization
  double lam = par[1];       // mean of x
  double f = 0;
  f = A * exp(x/lam); 
  return f;
}

//-------------------------------------------------------------------------
// This is our OBJECTIVE Function
// return NLL given a histogram and function

double calcNLL(TH1F* h, TF1* f){
  double nll=0;
  for (int i=1; i<=h->GetNbinsX(); i++){
    double x=h->GetBinCenter(i);
    int n=(int)(h->GetBinContent(i));
    double mu=f->Eval(x);
    if (mu<1e-10) mu=1e-10;    // avoid log(0) problems if we go outside a reasonable range!
    nll -= n * TMath::Log(mu) - mu  - TMath::LnGamma(n+1);
  }
  // cout << "nll "<< nll <<endl;
  return 2*nll;   // factor of 2 so the 1 sigma error contours follow the chi^2 convention
}


//-------------------------------------------------------------------------
// Minuit fcn: calculates value of the function to be minimized using
// the data and the model function
// This is the interface used to define our objective function.  We use
// only a subset of the input parameters below.
// npar: number of parameters
// par: array of parameter values
// f: the value of the objective function
// Minuit can also pass the gradient(deriv) of the objective function wrt the
// current parameters or flags that can trigger special operations (eg perform some initialization)

void fcn(int& npar, double* deriv, double& f, double par[], int flag){

  for (int i=0; i<npar; i++){
    fparam->SetParameter(i,par[i]);
  }

  f = calcNLL(hdata,fparam);
 
}

//-------------------------------------------------------------------------

int main(int argc, char **argv) {

  // This allows you to view ROOT-based graphics in your C++ program
  // If you don't want view graphics (eg just read/process/write data files), 
  // this line can be removed
  TApplication theApp("App", &argc, argv);

  TCanvas* canvas = new TCanvas();
  // ***************************************
  // not important for this example
  // Set a bunch of parameters to make the plot look nice
  canvas->SetFillColor(0);
  canvas->UseCurrentStyle();
  canvas->SetBorderMode(0);        
  canvas->SetFrameBorderMode(0);  
  //gROOT->SetStyle("Plain");
  canvas->UseCurrentStyle();
  gROOT->ForceStyle();

  //gStyle->SetOptStat(0);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetTitleSize(0.04);
  gStyle->SetTitleFont(42, "hxy");      // for histogram and axis titles
  gStyle->SetLabelFont(42, "xyz");      // for axis labels (values)
  gROOT->ForceStyle();
  // ***************************************


  // Use this to generate some random data 
  TRandom2 r(0);
  
  // create a histogram and fill it w/ randomly distributed data
  // based on an exponential PDF
  double xmin = 0.0;
  double xmax = 4.0;
  double lambda = 3.14159;
  int nentries = 1000;
  TH1F *hexp=new TH1F("hexp","exponential distribution;x;# events",50,xmin,xmax);

  for (int i=0; i<nentries; i++){
    hexp->Fill(r.Exp(lambda));  // generate random exponential distribution
  }
  
  
  // Initialize minuit, set initial values etc. of parameters.
  const int npar = 2;              // the number of parameters
 
  TMinuit minuit(npar);
  minuit.SetFCN(fcn);              // the fcn to be minized (objective fcn) 
  // this calculates chi^2 or NLL, it is not the parameterization of the data!

  // Make a TF1, we can use this to fit the data and plot the results
  TF1* myfunc = new TF1("myfunc", expPdf, xmin, xmax, npar);  
  double par[npar];               // the free parameters
  double stepSize[npar];          // (limiting) step sizes for parameters
  double minVal[npar];            // minimum/maximum bounds on parameters 
  double maxVal[npar];            
  TString parName[npar];          // Optional names for nicer output!

  // Initial parameters MUST be set by some means to get things started
  par[0] = hexp->GetMaximum();       // guesses for starting the fit
  par[1] = -2;                      
  stepSize[0] = TMath::Abs(par[0]*0.1);   // usually 10% of initial guess is OK for starting step size, YMMV
  stepSize[1] = TMath::Abs(par[1]*0.1);   // step sizes MUST be positive!
  minVal[0] = 0;      // if min and max values = 0, parameter is unbounded.
  maxVal[0] = 0;
  minVal[1] = 0; 
  maxVal[1] = 0;

  parName[0] = "A";       // let's give our fit parameters useful names
  parName[1] = "lambda";

  // initialize the parameters
  for (int i=0; i<npar; i++){
    minuit.DefineParameter(i, parName[i].Data(), 
			   par[i], stepSize[i], minVal[i], maxVal[i]);
  }

  // here we define the pointers to pass information to Minuit's fcn to calculate the
  // objective function
  // note: the use of global variables is discouraged in general, but for these
  // examples we'll let convenience and code simplicity outweigh the programming
  // style considerations.
  hdata=hexp;     // histogram to fit
  fparam=myfunc;  // our model

  // Do the minimization!
  minuit.Migrad();       // Minuit's best minimization algorithm

  // Get the result
  double outpar[npar], err[npar];
  for (int i=0; i<npar; i++){
    minuit.GetParameter(i,outpar[i],err[i]);
  }
  // run a minos error analysis
  // this perfoms a brute force scan around the minimum of the objective function
  // to get better estimates of the errors on the fit parameters
  // gMinuit->mnmnos();

  // store the fit parameters in our TF1, decorate our function
  myfunc->SetParameters(outpar);

  myfunc->SetLineStyle(1);             //  1 = solid, 2 = dashed, 3 = dotted
  myfunc->SetLineColor(1);             //  black (default)
  myfunc->SetLineWidth(2);

  myfunc->GetXaxis()->SetTitle("x");
  myfunc->GetYaxis()->SetTitle("f(x;#lambda)");

  // Plot the result.
  hdata->Draw("e");
  myfunc->Draw("same");

  // summarize the fitting results
  cout << "\n==========================\n"<<endl;
  double fmin, fedm, errdef;
  int npari, nparx, istat;  // see https://root.cern/doc/master/classTMinuit.html 
  minuit.mnstat(fmin, fedm, errdef, npari, nparx, istat);
  cout << "minimum of NLL = " << fmin << endl;
  cout << "fit status = " << istat << endl;
  cout << "best fit parameters\n" <<endl;
  for (int i=0; i<npar; ++i){
    cout << i << " : " << outpar[i] << " +- " << err[i] << endl;
  }

  cout << "\nTo exit, quit ROOT from the File menu of the plot (or use control-C)" << endl;
  theApp.SetIdleTimer(30,".q");  // set up a failsafe timer to end the program
  theApp.Run(true);
  canvas->Close();

  return 0;

}
