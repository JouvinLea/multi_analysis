#! /usr/bin/env python
# encoding: UTF-8
from astropy.table import Table
import numpy as np
import pylab as pt
from matplotlib.backends.backend_pdf import PdfPages
from gammapy.utils.energy import EnergyBounds,Energy
from method_fit import *
from method_plot import *
import yaml
import sys
pt.ion()
"""
./plot_spectra.py "config_crab.yaml"
plot le flux des differentes composantes utilisees pour fitter le on quand on estime le flux dans la source
"""

input_param=yaml.load(open(sys.argv[1]))
#Input param fit and source configuration
freeze_bkg=input_param["param_fit_morpho"]["freeze_bkg"]
param_fit = input_param["param_fit_morpho"]

source_name=input_param["general"]["source_name"]
name_method_fond = input_param["general"]["name_method_fond"]
image_size= input_param["general"]["image_size"]
for_integral_flux=input_param["exposure"]["for_integral_flux"] 
if freeze_bkg:
    name="_bkg_fix"
else:
    name="_bkg_free"


if param_fit["gauss_SgrA"]["fit"]:
    name += "_SgrA"
if param_fit["gauss_G0p9"]["fit"]:
    name += "_G0p9"
# Si on inverse LS et CS alors c est qu il y a les deux!
if param_fit["invert_CS_LS"]:
    if param_fit["invert_CC_LS"]:
        name += "_CS__central_gauss_LS"
    else:
        name += "_CS__LS_central_gauss"
else:
    if param_fit["Large scale"]["fit"]:
        name += "_LS"
    if param_fit["Gauss_to_CS"]["fit"]:
        name += "_CS"
    if param_fit["central_gauss"]["fit"]:
        name += "_central_gauss"
if param_fit["arc source"]["fit"]:
    name += "_arcsource"
if not param_fit["arc source"]["xpos_frozen"]:
    name += "_pos_free"

if param_fit["SgrB2"]["fit"]:
    name += "_SgrB2"
if param_fit["Large scale"]["fwhm_min"]:
    name += "_LS_fwhm_min_"+str(param_fit["Large scale"]["fwhm_min"])
if param_fit["Gauss_to_CS"]["fwhm_min"]:
    name += "_CS_fwhm_min_"+str(param_fit["Gauss_to_CS"]["fwhm_min"])

    
name += "_GC_source_fwhm_"+str(param_fit["gauss_SgrA"]["fwhm_init"])
if param_fit["Large scale"]["ellip_frozen"]:
    name += "_eLS_"+str(param_fit["Large scale"]["ellip_init"])
if param_fit["Large scale"]["fwhm_frozen"]:
    name += "_fwhmLS_"+str(param_fit["Large scale"]["fwhm_init"])
if param_fit["Gauss_to_CS"]["fwhm_frozen"]:
    name += "_fwhmCS_"+str(param_fit["Gauss_to_CS"]["fwhm_init"])

#Energy binning
energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"], input_param["energy binning"]["Emax"], input_param["energy binning"]["nbin"], 'TeV')
E_center=energy_bins.log_centers
energy_reco=[Energy(input_param["energy binning"]["Emin"],"TeV"),Energy(input_param["energy binning"]["Emax"],"TeV"), input_param["energy binning"]["nbin"]]

#outdir result and plot
config_name = input_param["general"]["config_name"]
outdir_result = make_outdir_filesresult(source_name, name_method_fond,config_name,image_size,for_integral_flux=False,ereco=energy_reco)
outdir_plot = make_outdir_plot(source_name, name_method_fond,config_name,image_size,for_integral_flux=False,ereco=energy_reco)


#imax: until which energy bin we want to plot
imax=input_param["param_plot"]["imax"]

for i_E, E in enumerate(energy_bins[0:imax]):
    E1 = energy_bins[i_E].value
    E2 = energy_bins[i_E+1].value
    pdf=PdfPages(outdir_plot+"/param_by_step_" +name+"_"+ str("%.2f"%E1) + "_" + str("%.2f"%E2) + "_TeV.pdf")
    filename_table_result=outdir_result + "/morphology_fit_result_" + name + "_" + str("%.2f" % E1) + "_" + str(
        "%.2f" % E2) + "_TeV.txt"
    filename_covar_result=outdir_result + "/morphology_fit_covar_" + name + "_" + str("%.2f" % E1) + "_" + str(
        "%.2f" % E2) + "_TeV.txt"
    table=Table.read(filename_table_result, format="ascii")
    table_covar=Table.read(filename_covar_result, format="ascii")
    for i,model_name in enumerate(table.colnames):
        if ((model_name=="step") | (model_name=="dof")):
            continue
        elif (model_name=="statval"):
            pt.figure()
            i_ok=np.where(table.filled(-1000)[model_name]!=-1000.0)[0]
            pt.plot(table["step"][i_ok], table[model_name][i_ok], "*",color="blue")
            pt.xlim(0,7)
            pt.xlabel("step")
            pt.ylabel(model_name)
            pdf.savefig()
            TS=table[model_name][:-1]-table[model_name][1:]
            pt.figure()
            pt.plot(table["step"][1:], TS, "*",color="blue")
            pt.xlim(0,7)
            pt.xlabel("step")
            pt.ylabel("TS")
            pt.title("Evolution du TS pour les differents steps")
            pdf.savefig()
            pt.figure()
            pt.plot(table["step"][4:], TS[3:], "*",color="blue")
            pt.xlim(0,7)
            pt.xlabel("step")
            pt.ylabel("TS")
            pt.title("Evolution du TS pour les derniers steps")
            pdf.savefig()

        else:
            pt.figure()
            i_ok=np.where(table.filled(-1000)[model_name]!=-1000.0)[0]
            #import IPython; IPython.embed()
            #table_covar[name+"_max"][np.where(table_covar[name+"_max"]==-1000.0)]=0
            #table_covar[name+"_min"][np.where(table_covar[name+"_min"]==-1000.0)]=0
            pt.plot(table["step"][i_ok], table[model_name][i_ok], "*",color="blue")
            pt.errorbar(table["step"][i_ok], table[model_name][i_ok], yerr=[-table_covar[model_name+"_min"][i_ok],table_covar[model_name+"_max"][i_ok]],linestyle="None", color="blue")
            pt.xlim(0,7)
            pt.ylim(ymin=0)
            pt.xlabel("step")
            pt.ylabel(model_name)
            pdf.savefig()
        print model_name
    pdf.close()





