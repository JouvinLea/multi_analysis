
#! /usr/bin/env python
from sherpa.astro.ui import *
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from astropy.table import join
import astropy.units as u
from IPython.core.display import Image
from gammapy.image import SkyImageList, SkyImage
from gammapy.utils.energy import EnergyBounds, Energy
import astropy.units as u
import pylab as pt
from gammapy.background import fill_acceptance_image
from gammapy.utils.energy import EnergyBounds,Energy
from astropy.coordinates import Angle
from astropy.units import Quantity
import numpy as np
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from method_fit import *
from matplotlib.backends.backend_pdf import PdfPages
from gammapy.detect import compute_ts_image
from astropy.convolution import Gaussian2DKernel
import yaml
import sys

pt.ion()

"""
./estimation_sourceflux.py "config_crab.yaml"
Estimation du flux du source model a partir de la psf et de l exposure: on=bkg+psf(model*exposure)
"""

input_param = yaml.load(open(sys.argv[1]))
image_size= input_param["general"]["image_size"]
# Input param fit and source configuration
# Sur quelle taille de la carte on fait le fit
freeze_bkg = input_param["param_fit_morpho"]["freeze_bkg"]
source_name = input_param["general"]["source_name"]
name_method_fond = input_param["general"]["name_method_fond"]
if freeze_bkg:
    name = "_bkg_fix"
else:
    name = "_bkg_free"
for_integral_flux=input_param["exposure"]["for_integral_flux"]
# Energy binning
energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"],
                                             input_param["energy binning"]["Emax"],
                                             input_param["energy binning"]["nbin"], 'TeV')
energy_centers = energy_bins.log_centers
energy_reco=[Energy(input_param["energy binning"]["Emin"],"TeV"),Energy(input_param["energy binning"]["Emax"],"TeV"), input_param["energy binning"]["nbin"]]

# outdir data and result
config_name = input_param["general"]["config_name"]
outdir_data = make_outdir_data(source_name, name_method_fond,config_name,image_size,for_integral_flux=False,ereco=energy_reco)
outdir_result = make_outdir_filesresult(source_name, name_method_fond,config_name,image_size,for_integral_flux=False,ereco=energy_reco)
outdir_plot = make_outdir_plot(source_name, name_method_fond,config_name,image_size,for_integral_flux=False,ereco=energy_reco)
outdir_profiles = make_outdir_profile(source_name, name_method_fond,config_name,image_size,for_integral_flux=False,ereco=energy_reco)

# Pour pouvoir definir la gaussienne centre sur la source au centre des cartes en general
E1 = energy_bins[0].value
E2 = energy_bins[1].value
on = SkyImageList.read(outdir_data + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["counts"]

if "l_gal" in input_param["param_SgrA"]["sourde_name_skycoord2"]:
    source_center =  SkyCoord(input_param["param_SgrA"]["sourde_name_skycoord2"]["l_gal"],
                                  input_param["param_SgrA"]["sourde_name_skycoord2"]["b_gal"], unit='deg',
                                  frame="galactic").icrs
else:
    source_center = SkyCoord.from_name(input_param["general"]["sourde_name_skycoord"])


param_fit = input_param["param_fit_morpho"]
if param_fit["Em_gal"]:
    name += "_Em_gal"
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
                                     
                                     
residus_l=list()
residus_err_l=list()
residus_b=list()
residus_err_b=list()
residus_l_SgrA_G0p9_arc_source=list()
for i_E, E in enumerate(energy_bins[0:4]):
#for i_E, E in enumerate(energy_bins[0:1]):
    E1 = energy_bins[i_E].value
    E2 = energy_bins[i_E + 1].value
    energy_band = Energy([E1 , E2], energy_bins.unit)
    print "energy band: ", E1, " TeV- ", E2, "TeV"
    # load Data
    on = SkyImageList.read(outdir_data + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["counts"]
    on.write(outdir_data + "/on_maps" + str(E1) + "_" + str(E2) + "_TeV.fits", clobber=True)
    data = fits.open(outdir_data + "/on_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")
    load_image(1, data)
    # load exposure model
    exposure = make_exposure_model(outdir_data, E1, E2)
    # load bkg model
    bkg = make_bkg_model(outdir_data, E1, E2, freeze_bkg)
    # load psf model
    source_center_SgrA = SkyCoord(input_param["param_SgrA"]["sourde_name_skycoord2"]["l_gal"],
                                  input_param["param_SgrA"]["sourde_name_skycoord2"]["b_gal"], unit='deg',
                                  frame="galactic")
    psf_SgrA = make_psf_model(outdir_data, E1, E2, on, source_name)
    #psf_G0p9 = make_psf_model(outdir_data, E1, E2, on, "G0.9")
    """
    # load CS model
    CS = make_CS_model(outdir_data, on, None, param_fit["CS"]["ampl_frozen"],
                       param_fit["CS"]["threshold_map"])
    # modele gauss pour sgrA centre sur SgrA
    #source_center_SgrA = SkyCoord.from_name(input_param["param_SgrA"]["sourde_name_skycoord"])
    source_center_SgrA = SkyCoord(input_param["param_SgrA"]["sourde_name_skycoord2"]["l_gal"],
                                  input_param["param_SgrA"]["sourde_name_skycoord2"]["b_gal"], unit='deg',
                                  frame="galactic")
    xpos_SgrA, ypos_SgrA = skycoord_to_pixel(source_center_SgrA, on.wcs)
    xpos_GC, ypos_GC = skycoord_to_pixel(source_center_SgrA, on.wcs)
    xpos_SgrA += 0.5
    ypos_SgrA += 0.5
    mygaus_SgrA = source_punctual_model(param_fit["gauss_SgrA"]["name"], param_fit["gauss_SgrA"]["fwhm_init"],
                                        param_fit["gauss_SgrA"]["fwhm_frozen"], param_fit["gauss_SgrA"]["ampl_init"],
                                        param_fit["gauss_SgrA"]["ampl_frozen"], xpos_SgrA,
                                        param_fit["gauss_SgrA"]["xpos_frozen"],
                                        ypos_SgrA, param_fit["gauss_SgrA"]["ypos_frozen"], ampl_max=1e-4)
    # modele gauss pour G0p9 centre sur G0p9
    source_center_G0p9 = SkyCoord(input_param["param_G0p9"]["sourde_name_skycoord"]["l_gal"],
                                  input_param["param_G0p9"]["sourde_name_skycoord"]["b_gal"], unit='deg',
                                  frame="galactic")
    xpos_G0p9, ypos_G0p9 = skycoord_to_pixel(source_center_G0p9, on.wcs)
    xpos_G0p9 += 0.5
    ypos_G0p9 += 0.5
    mygaus_G0p9 = source_punctual_model(param_fit["gauss_G0p9"]["name"], param_fit["gauss_G0p9"]["fwhm_init"],
                                        param_fit["gauss_G0p9"]["fwhm_frozen"], param_fit["gauss_G0p9"]["ampl_init"],
                                        param_fit["gauss_G0p9"]["ampl_frozen"], xpos_G0p9,
                                        param_fit["gauss_G0p9"]["xpos_frozen"],
                                        ypos_G0p9, param_fit["gauss_G0p9"]["ypos_frozen"], ampl_max=1e-4)

    # modele asymetric large scale gauss centre sur SgrA
    param_fit["Large scale"]["fwhm_init"] = None if param_fit["Large scale"]["fwhm_init"] == 'None' else param_fit["Large scale"]["fwhm_init"]
    param_fit["Large scale"]["fwhm_min"] = None if param_fit["Large scale"]["fwhm_min"] == 'None' else param_fit["Large scale"]["fwhm_min"]
    param_fit["Large scale"]["ampl_init"] = None if param_fit["Large scale"]["ampl_init"] == 'None' else param_fit["Large scale"]["ampl_init"]
    Large_Scale = source_NormGauss2D(param_fit["Large scale"]["name"], param_fit["Large scale"]["fwhm_init"],
                              param_fit["Large scale"]["fwhm_frozen"], param_fit["Gauss_to_CS"]["ampl_init"],
                              param_fit["Large scale"]["ampl_frozen"], xpos_GC,
                              param_fit["Large scale"]["xpos_frozen"],
                              ypos_GC, param_fit["Large scale"]["ypos_frozen"],ellep_fit=True,
                              ellep_init=param_fit["Large scale"]["ellip_init"],
                                     ellep_frozen=param_fit["Large scale"]["ellip_frozen"], ampl_max=None, fwhm_min=param_fit["Large scale"]["fwhm_min"])

    # Modele large gaussienne  multiplie avec CS centre sur SgrA
    param_fit["Gauss_to_CS"]["fwhm_init"] = None if param_fit["Gauss_to_CS"]["fwhm_init"] == 'None' else param_fit["Gauss_to_CS"]["fwhm_init"]
    param_fit["Gauss_to_CS"]["fwhm_min"] = None if param_fit["Gauss_to_CS"]["fwhm_min"] == 'None' else param_fit["Gauss_to_CS"]["fwhm_min"]
    param_fit["Gauss_to_CS"]["ampl_init"] = None if param_fit["Gauss_to_CS"]["ampl_init"] == 'None' else param_fit["Gauss_to_CS"]["ampl_init"]
    gaus_CS = source_Gauss2D(param_fit["Gauss_to_CS"]["name"], param_fit["Gauss_to_CS"]["fwhm_init"],
                          param_fit["Gauss_to_CS"]["fwhm_frozen"], param_fit["Gauss_to_CS"]["ampl_init"],
                          param_fit["Gauss_to_CS"]["ampl_frozen"], xpos_GC, param_fit["Gauss_to_CS"]["xpos_frozen"],
                             ypos_GC, param_fit["Gauss_to_CS"]["ypos_frozen"],  ampl_max=None, fwhm_min=param_fit["Gauss_to_CS"]["fwhm_min"])

    # Modele symetric central gauss centre sur SgrA
    central_gauss = source_NormGauss2D(param_fit["central_gauss"]["name"], param_fit["central_gauss"]["fwhm_init"],
                                param_fit["central_gauss"]["fwhm_frozen"], param_fit["central_gauss"]["ampl_init"],
                                param_fit["central_gauss"]["ampl_frozen"], xpos_GC,
                                param_fit["central_gauss"]["xpos_frozen"],
                                ypos_GC, param_fit["central_gauss"]["ypos_frozen"])
    
    #Arc_source
    source_center_arcsource = SkyCoord(param_fit["arc source"]["l"],
                       param_fit["arc source"]["b"], unit='deg', frame="galactic")
    xpos_arcsource, ypos_arcsource = skycoord_to_pixel(source_center_arcsource, on.wcs)
    arc_source=source_NormGauss2D(param_fit["arc source"]["name"], param_fit["arc source"]["fwhm_init"],
                                       param_fit["arc source"]["fwhm_frozen"], param_fit["arc source"]["ampl_init"],
                                       param_fit["arc source"]["ampl_frozen"], xpos_arcsource, param_fit["arc source"]["xpos_frozen"],
                          ypos_arcsource, param_fit["arc source"]["ypos_frozen"])
    #Gauss SgrB2
    source_center_sgrB2 = SkyCoord(param_fit["SgrB2"]["l"],
                       param_fit["SgrB2"]["b"], unit='deg', frame="galactic")
    xpos_sgrB2, ypos_sgrB2 = skycoord_to_pixel(source_center_sgrB2, on.wcs)
    sgrB2=source_NormGauss2D(param_fit["SgrB2"]["name"], param_fit["SgrB2"]["fwhm_init"],
                                       param_fit["SgrB2"]["fwhm_frozen"], param_fit["SgrB2"]["ampl_init"],
                                       param_fit["SgrB2"]["ampl_frozen"], xpos_sgrB2, param_fit["SgrB2"]["xpos_frozen"],
                          ypos_sgrB2, param_fit["SgrB2"]["ypos_frozen"])
    """
    #region of inerest
    """
    pix_deg = on.to_image_hdu().header["CDELT2"]
    lat=1.6/ pix_deg#Pour aller a plus et -0.8 as did Anne
    lon=4 / pix_deg#Pour aller a plus ou moins 2deg as did Anne
    x_pix_SgrA=skycoord_to_pixel(source_center_SgrA, on.wcs)[0]
    y_pix_SgrA=skycoord_to_pixel(source_center_SgrA, on.wcs)[1]
    name_interest = "box(" + str(x_pix_SgrA) + "," + str(y_pix_SgrA) + "," + str(lon) + "," + str(lat) +")"
    #name_interest = "box(" + str(x_pix_SgrA) + "," + str(y_pix_SgrA) + "," + str(150) + "," + str(50) +")"
    #name_interest = "box(" + str(x_pix_SgrA) + "," + str(y_pix_SgrA) + "," + str(10) + "," + str(10) +")"
    #notice2d(name_interest)
    ignore2d(name_interest)
    
    #ignore region in a box that mask J1734-303
    source_J1745_303 = SkyCoord(358.76, -0.6, unit='deg', frame="galactic")
    source_J1745_303_xpix, source_J1745_303_ypix = skycoord_to_pixel(source_J1745_303, on.wcs)
    width=100
    height=80
    name_region = "box(" + str(source_J1745_303_xpix+20) + "," + str(source_J1745_303_ypix-20) + "," + str(width) + "," + str(height) +")"
    ignore2d(name_region)
    """
    #name_interest=region_interest(source_center, on, 250, 250)
    #notice2d(name_interest)
    #name_region1 = "circle(107,178,15)"
    #name_region2 = "box(125,80,250,60)"
    name_region1 = "circle(107,178,20)"
    name_region2 = "box(125,80,250,80)"
    ignore2d(name_region1)
    ignore2d(name_region2)
    #ignore2d("ds9.reg")
    set_stat("cstat")
    set_method("neldermead")


    if param_fit["Em_gal"]:
        #EmGal=make_EG_model(outdir_data, on,1e-8, None)
        EmGal=make_EG_model(outdir_data, on,1, None)
        #EmGal=make_EG_model(outdir_data, on,2e-4, True)
        #EmGal=make_EG_model(outdir_data, on,2e-10, True)
        #EmGal=make_EG_model(outdir_data, on,1.83937 , True)
        model = bkg+psf_SgrA(exposure*EmGal)
        #model = psf_SgrA(exposure*EmGal)
    else:
        model = bkg
    set_full_model(model)
    fit()
    result= get_fit_results()
    #import IPython; IPython.embed()
    shape = np.shape(on.data)
    mask = get_data().mask.reshape(shape)
    map_data=SkyImage.empty_like(on)
    model_map =SkyImage.empty_like(on)
    resid =SkyImage.empty_like(on)
    map_data.data = get_data().y.reshape(shape) * mask
    model_map.data = get_model()(get_data().x0, get_data().x1).reshape(shape) * mask
    resid.data = map_data.data - model_map.data
    pt.figure(1)
    resid.plot(add_cbar=True)
    pt.savefig("resid_Em_gal.png")
    print("Counts data: "+str(map_data.data.sum()))
    print("Counts model: "+str(model_map.data.sum()))

        
