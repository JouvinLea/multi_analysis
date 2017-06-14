import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt
from gammapy.utils.energy import EnergyBounds, Energy
from gammapy.image import SkyImageList
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
import yaml
import sys
from method_fit import *

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
#if param_fit["Large scale"]["fwhm_min"]:
name += "_LS_fwhm_min_"+str(param_fit["Large scale"]["fwhm_min"])+"_init_"+str(param_fit["Large scale"]["fwhm_init"])
#if param_fit["Gauss_to_CS"]["fwhm_min"]:
name += "_CS_fwhm_min_"+str(param_fit["Gauss_to_CS"]["fwhm_min"])+"_init_"+str(param_fit["Gauss_to_CS"]["fwhm_init"])

  
name += "_GC_source_fwhm_"+str(param_fit["gauss_SgrA"]["fwhm_init"])     
if param_fit["Large scale"]["ellip_frozen"]:
    name += "_eLS_"+str(param_fit["Large scale"]["ellip_init"])
if param_fit["Large scale"]["fwhm_frozen"]:
    name += "_fwhmLS_"+str(param_fit["Large scale"]["fwhm_init"])
if param_fit["Gauss_to_CS"]["fwhm_frozen"]:
    name += "_fwhmCS_"+str(param_fit["Gauss_to_CS"]["fwhm_init"])
if param_fit["central_gauss"]["fwhm_frozen"]:
    name += "_fwhmCC_"+str(param_fit["central_gauss"]["fwhm_init"])              
config_name = input_param["general"]["config_name"]
energy_reco=[Energy(input_param["energy binning"]["Emin"],"TeV"),Energy(input_param["energy binning"]["Emax"],"TeV"), input_param["energy binning"]["nbin"]]


outdir_data = make_outdir_data(source_name, name_method_fond,config_name,image_size,for_integral_flux=False,ereco=energy_reco)
directory = make_outdir_filesresult(source_name, name_method_fond,config_name,image_size,for_integral_flux=False,ereco=energy_reco)

#directory="/Users/jouvin/Desktop/these/test_Gammapy/script/ash_stereo_thsq64/Image_GC_bkg_coszenbinning_zen_0_34_49_61_72_26binE/binE_1_min_0.5_max_100.0_size_image_250_pix_exposure_flux_diff/files_result"
#outdir_data="/Users/jouvin/Desktop/these/test_Gammapy/script/ash_stereo_thsq64/Image_GC_bkg_coszenbinning_zen_0_34_49_61_72_26binE/binE_1_min_0.5_max_100.0_size_image_250_pix_exposure_flux_diff"
energy_bins=EnergyBounds.equal_log_spacing(0.5,100,1,"TeV")
pix_to_deg=0.02
for i, E in enumerate(energy_bins[:-1]):
    E1=energy_bins[i].value
    E2=energy_bins[i+1].value
    print("Energy band:"+str("%.2f" % E1)+"-"+str("%.2f" % E2)+" TeV")
    filename=directory+"/morphology_fit_result_"+name+"_"+str("%.2f" % E1)+"_"+str("%.2f" % E2)+"_TeV.txt"
    filename_err=directory+"/morphology_fit_covar_"+name+"_"+str("%.2f" % E1)+"_"+str("%.2f" % E2)+"_TeV.txt"
    t=Table.read(filename, format="ascii")
    t_err=Table.read(filename_err, format="ascii")
    for istep,step in enumerate(t["step"][:-1]):
        sigma=np.sqrt(np.fabs((t["statval"][istep+1]-t["statval"][istep])))
        print("step: "+str(step)+", detection a "+str(sigma)+" sigma")
    ifinal_step=len(t)-1
    CS_fwwhm=t[ifinal_step]["Gauss*Templ_CS.fwhm"]*pix_to_deg/2.35
    CS_fwwhm_err=t_err[ifinal_step]["Gauss*Templ_CS.fwhm_min"]*pix_to_deg/2.35
    LS_fwwhm=t[ifinal_step]["Asym Large Scale.fwhm"]*pix_to_deg/2.35
    LS_fwwhm_err=t_err[ifinal_step]["Asym Large Scale.fwhm_min"]*pix_to_deg/2.35
    LS_ellip=t[ifinal_step]["Asym Large Scale.ellip"]
    LS_ellip_err=t_err[ifinal_step]["Asym Large Scale.ellip_min"]
    sigma_x=LS_ellip*LS_fwwhm
    sigma_y=(1-LS_ellip)*LS_fwwhm
    #Err variable correle: Var(f(x,y))=(df/dx*dx)**2+(df/dy*dy)**2+2*df/dx*df/dy*dx*dy
    sigma_x_err=np.sqrt(((LS_ellip)*LS_fwwhm_err)**2+(LS_fwwhm*LS_ellip_err)**2+2*(LS_fwwhm*(1-LS_ellip)*LS_fwwhm_err*LS_ellip_err))
    sigma_y_err=np.sqrt(((1-LS_ellip)*LS_fwwhm_err)**2+(LS_fwwhm*LS_ellip_err)**2+2*(LS_fwwhm*(1-LS_ellip)*LS_fwwhm_err*LS_ellip_err))
    CC_fwwhm=t[ifinal_step]["Central Component.fwhm"]*pix_to_deg/2.35
    CC_fwwhm_err=t_err[ifinal_step]["Central Component.fwhm_min"]*pix_to_deg/2.35
    print("Gauss*Templ_CS.fwhm=",CS_fwwhm," +/- ",CS_fwwhm_err)
    print("LS fwhm=",LS_fwwhm," +/- ",LS_fwwhm_err)
    print("LS ellipticity=",LS_ellip," +/- ",LS_ellip_err)
    print("LS sigmax=",sigma_x," +/- ",sigma_x_err)
    print("LS sigmay=",sigma_y," +/- ",sigma_y_err)
    print("Central Component =",CC_fwwhm," +/- ",CC_fwwhm_err)
    if 'Arc source.xpos' in t.colnames:
        on = SkyImageList.read(outdir_data + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["counts"]
        coord_arc_source=pixel_to_skycoord(t["Arc source.xpos"][ifinal_step],t["Arc source.ypos"][ifinal_step],on.wcs)
        print coord_arc_source
    
