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
from Em_gal import UniformGaussianPlane

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
if param_fit["Em_gal_hgps"]["use"] & param_fit["Em_gal_ana"]["use"]:
    sys.exit('You can not ask for two different Emgal model emission')
if param_fit["Em_gal_hgps"]["use"]:
    name += "_Em_gal_hgps"
if param_fit["Em_gal_ana"]["use"]:
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
    psf_SgrA = make_psf_model(outdir_data, E1, E2, on, source_name)
    psf_G0p9 = make_psf_model(outdir_data, E1, E2, on, "G0.9")
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
                                        ypos_SgrA, param_fit["gauss_SgrA"]["ypos_frozen"], ampl_max=param_fit["gauss_SgrA"]["ampl_max"])
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
                                        ypos_G0p9, param_fit["gauss_G0p9"]["ypos_frozen"], param_fit["gauss_G0p9"]["ampl_max"])

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
                                     ellep_frozen=param_fit["Large scale"]["ellip_frozen"], ampl_max=param_fit["Large scale"]["ampl_max"], fwhm_min=param_fit["Large scale"]["fwhm_min"])

    # Modele large gaussienne  multiplie avec CS centre sur SgrA
    param_fit["Gauss_to_CS"]["fwhm_init"] = None if param_fit["Gauss_to_CS"]["fwhm_init"] == 'None' else param_fit["Gauss_to_CS"]["fwhm_init"]
    param_fit["Gauss_to_CS"]["fwhm_min"] = None if param_fit["Gauss_to_CS"]["fwhm_min"] == 'None' else param_fit["Gauss_to_CS"]["fwhm_min"]
    param_fit["Gauss_to_CS"]["ampl_init"] = None if param_fit["Gauss_to_CS"]["ampl_init"] == 'None' else param_fit["Gauss_to_CS"]["ampl_init"]
    gaus_CS = source_Gauss2D(param_fit["Gauss_to_CS"]["name"], param_fit["Gauss_to_CS"]["fwhm_init"],
                          param_fit["Gauss_to_CS"]["fwhm_frozen"], param_fit["Gauss_to_CS"]["ampl_init"],
                          param_fit["Gauss_to_CS"]["ampl_frozen"], xpos_GC, param_fit["Gauss_to_CS"]["xpos_frozen"],
                             ypos_GC, param_fit["Gauss_to_CS"]["ypos_frozen"],  ampl_max=param_fit["Gauss_to_CS"]["ampl_max"], fwhm_min=param_fit["Gauss_to_CS"]["fwhm_min"])

    # Modele symetric central gauss centre sur SgrA
    param_fit["central_gauss"]["fwhm_init"] = None if param_fit["central_gauss"]["fwhm_init"] == 'None' else param_fit["central_gauss"]["fwhm_init"]
    central_gauss = source_punctual_model(param_fit["central_gauss"]["name"], param_fit["central_gauss"]["fwhm_init"],
                                param_fit["central_gauss"]["fwhm_frozen"], param_fit["central_gauss"]["ampl_init"],
                                param_fit["central_gauss"]["ampl_frozen"], xpos_GC,
                                param_fit["central_gauss"]["xpos_frozen"],
                                       ypos_GC, param_fit["central_gauss"]["ypos_frozen"], ampl_max=param_fit["central_gauss"]["ampl_max"])
    #Arc_source
    source_center_arcsource = SkyCoord(param_fit["arc source"]["l"],
                       param_fit["arc source"]["b"], unit='deg', frame="galactic")
    xpos_arcsource, ypos_arcsource = skycoord_to_pixel(source_center_arcsource, on.wcs)
    arc_source=source_punctual_model(param_fit["arc source"]["name"], param_fit["arc source"]["fwhm_init"],
                                       param_fit["arc source"]["fwhm_frozen"], param_fit["arc source"]["ampl_init"],
                                       param_fit["arc source"]["ampl_frozen"], xpos_arcsource, param_fit["arc source"]["xpos_frozen"],
                                  ypos_arcsource, param_fit["arc source"]["ypos_frozen"],ampl_max=param_fit["arc source"]["ampl_max"])
    #Gauss SgrB2
    source_center_sgrB2 = SkyCoord(param_fit["SgrB2"]["l"],
                       param_fit["SgrB2"]["b"], unit='deg', frame="galactic")
    xpos_sgrB2, ypos_sgrB2 = skycoord_to_pixel(source_center_sgrB2, on.wcs)
    sgrB2=source_punctual_model(param_fit["SgrB2"]["name"], param_fit["SgrB2"]["fwhm_init"],
                                       param_fit["SgrB2"]["fwhm_frozen"], param_fit["SgrB2"]["ampl_init"],
                                       param_fit["SgrB2"]["ampl_frozen"], xpos_sgrB2, param_fit["SgrB2"]["xpos_frozen"],
                             ypos_sgrB2, param_fit["SgrB2"]["ypos_frozen"],ampl_max=param_fit["SgrB2"]["ampl_max"])
    """
    # Modele symetric central gauss centre sur SgrA
    param_fit["central_gauss"]["fwhm_init"] = None if param_fit["central_gauss"]["fwhm_init"] == 'None' else param_fit["central_gauss"]["fwhm_init"]
    central_gauss = source_NormGauss2D(param_fit["central_gauss"]["name"], param_fit["central_gauss"]["fwhm_init"],
                                param_fit["central_gauss"]["fwhm_frozen"], param_fit["central_gauss"]["ampl_init"],
                                param_fit["central_gauss"]["ampl_frozen"], xpos_GC,
                                param_fit["central_gauss"]["xpos_frozen"],
                                       ypos_GC, param_fit["central_gauss"]["ypos_frozen"], ampl_max=param_fit["central_gauss"]["ampl_max"])
    #Arc_source
    source_center_arcsource = SkyCoord(param_fit["arc source"]["l"],
                       param_fit["arc source"]["b"], unit='deg', frame="galactic")
    xpos_arcsource, ypos_arcsource = skycoord_to_pixel(source_center_arcsource, on.wcs)
    arc_source=source_NormGauss2D(param_fit["arc source"]["name"], param_fit["arc source"]["fwhm_init"],
                                       param_fit["arc source"]["fwhm_frozen"], param_fit["arc source"]["ampl_init"],
                                       param_fit["arc source"]["ampl_frozen"], xpos_arcsource, param_fit["arc source"]["xpos_frozen"],
                                  ypos_arcsource, param_fit["arc source"]["ypos_frozen"],ampl_max=param_fit["arc source"]["ampl_max"])
    #Gauss SgrB2
    source_center_sgrB2 = SkyCoord(param_fit["SgrB2"]["l"],
                       param_fit["SgrB2"]["b"], unit='deg', frame="galactic")
    xpos_sgrB2, ypos_sgrB2 = skycoord_to_pixel(source_center_sgrB2, on.wcs)
    sgrB2=source_NormGauss2D(param_fit["SgrB2"]["name"], param_fit["SgrB2"]["fwhm_init"],
                                       param_fit["SgrB2"]["fwhm_frozen"], param_fit["SgrB2"]["ampl_init"],
                                       param_fit["SgrB2"]["ampl_frozen"], xpos_sgrB2, param_fit["SgrB2"]["xpos_frozen"],
                             ypos_sgrB2, param_fit["SgrB2"]["ypos_frozen"],ampl_max=param_fit["SgrB2"]["ampl_max"])
    """
    #region of inerest
    pix_deg = on.to_image_hdu().header["CDELT2"]
    lat=1.6/ pix_deg#Pour aller a plus et -0.8 as did Anne
    lon=4 / pix_deg#Pour aller a plus ou moins 2deg as did Anne
    x_pix_SgrA=skycoord_to_pixel(source_center_SgrA, on.wcs)[0]
    y_pix_SgrA=skycoord_to_pixel(source_center_SgrA, on.wcs)[1]
    name_interest = "box(" + str(x_pix_SgrA) + "," + str(y_pix_SgrA) + "," + str(lon) + "," + str(lat) +")"
    #name_interest = "box(" + str(x_pix_SgrA) + "," + str(y_pix_SgrA) + "," + str(150) + "," + str(50) +")"
    #name_interest = "box(" + str(x_pix_SgrA) + "," + str(y_pix_SgrA) + "," + str(10) + "," + str(10) +")"
    notice2d(name_interest)

    #ignore region in a box that mask J1734-303
    source_J1745_303 = SkyCoord(358.76, -0.6, unit='deg', frame="galactic")
    source_J1745_303_xpix, source_J1745_303_ypix = skycoord_to_pixel(source_J1745_303, on.wcs)
    width=100
    height=80
    name_region = "box(" + str(source_J1745_303_xpix+20) + "," + str(source_J1745_303_ypix-20) + "," + str(width) + "," + str(height) +")"
    ignore2d(name_region)

    set_stat("cstat")
    set_method("neldermead")

    list_src = [1e-8*psf_SgrA(exposure*mygaus_SgrA)]
    #list_src = [psf_SgrA(exposure*mygaus_SgrA)]
    if param_fit["gauss_G0p9"]["fit"]:
        list_src.append(1e-8*psf_G0p9(exposure*mygaus_G0p9))
        #list_src.append(psf_G0p9(exposure*mygaus_G0p9))
    # Si on inverse LS et CS alors c est qu il y a les deux!
    if param_fit["invert_CS_LS"]:
        list_src.append(psf_SgrA(1e-8*exposure*gaus_CS * CS))
        if param_fit["invert_CC_LS"]:
            list_src.append(psf_SgrA(1e-8*exposure*central_gauss))    
            list_src.append(psf_SgrA(1e-8*exposure*Large_Scale))
        else:    
            list_src.append(psf_SgrA(1e-8*exposure*Large_Scale))
            list_src.append(psf_SgrA(1e-8*exposure*central_gauss))
    else:
        if param_fit["Large scale"]["fit"]:
            list_src.append(psf_SgrA(1e-8*exposure*Large_Scale))
        if param_fit["Gauss_to_CS"]["fit"]:
            list_src.append(psf_SgrA(1e-8*exposure*gaus_CS * CS))
        if param_fit["central_gauss"]["fit"]:
            list_src.append(psf_SgrA(1e-8*exposure*central_gauss))
    if param_fit["arc source"]["fit"]:
        list_src.append(psf_SgrA(1e-8*exposure*arc_source))
    if param_fit["SgrB2"]["fit"]:
        list_src.append(psf_SgrA(1e-8*exposure*sgrB2))

    #Emgal
    if param_fit["Em_gal_hgps"]["use"]:
        EmGal=make_EG_model(outdir_data, on,param_fit["Em_gal_hgps"]["ampl"], True)
        model = bkg+psf_SgrA(exposure*EmGal)
    elif param_fit["Em_gal_ana"]["use"]:
        EmGal=UniformGaussianPlane(on.wcs)
        res = on.wcs.wcs_world2pix(np.array([0.,param_fit["Em_gal_ana"]["bpos"]],ndmin=2),1)
        xpix,ypix = res[0]
        EmGal.ypos=ypix
        EmGal.ypos.freeze()
        EmGal.thick=param_fit["Em_gal_ana"]["thick"]      
        EmGal.thick.freeze()
        EmGal.ampl= param_fit["Em_gal_ana"]["ampl"]
        EmGal.ampl.freeze()
        model = bkg+1e-8*psf_SgrA(exposure*EmGal)    
    else:
        model = bkg
    set_full_model(model)
    
    pdf_lat=PdfPages(outdir_profiles+"/profiles_lattitude_"+name+"_" + str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.pdf")
    pdf_lon=PdfPages(outdir_profiles+"/profiles_longitude_"+name+"_" + str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.pdf")
    for i_src, src in enumerate(list_src):
        model += src
        set_full_model(model)
        fit()
        #import IPython; IPython.embed()
        result = get_fit_results()
        if i_src==0:
            table_models = result_table_CG(result, int(i_src))
        else:
            table_models = join(table_models.filled(-1000), result_table_CG(result, int(i_src)), join_type='outer')
        covar()
        covar_res = get_covar_results()
        # conf()
        # covar_res= get_conf_results()
        if i_src==0:
            table_covar = covar_table_CG(covar_res, int(i_src))
        else:
            table_covar = join(table_covar.filled(0), covar_table_CG(covar_res, int(i_src)), join_type='outer')
        
        #save_resid(outdir_result + "/residual_morpho_step_" + str(i_src) + "_"+ name + "_"  + str("%.2f" % E1) + "_" + str(
       #     "%.2f" % E2) + "_TeV.fits", clobber=True)
        # import IPython; IPython.embed()
        # Profil lattitude et longitude
        #import IPython; IPython.embed()
        shape = np.shape(on.data)
        mask = get_data().mask.reshape(shape)
        map_data=SkyImage.empty_like(on)
        model_map =SkyImage.empty_like(on)
        resid =SkyImage.empty_like(on)
        exp_map=SkyImage.empty_like(on)
        map_data.data = get_data().y.reshape(shape) * mask
        model_map.data = get_model()(get_data().x0, get_data().x1).reshape(shape) * mask
        exp_map.data= np.ones(map_data.data.shape)* mask
        #import IPython; IPython.embed()
        resid.data = map_data.data - model_map.data
        resid.write(outdir_result + "/residual_morpho_et_flux_step_" + str(i_src) + "_"+ name + "_"  + str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.fits", clobber=True)
    if param_fit["Em_gal_hgps"]["use"]:
        model=  bkg + psf_SgrA(exposure*EmGal)+psf_SgrA(1e-8*exposure*mygaus_SgrA) + psf_G0p9(1e-8*exposure*mygaus_G0p9)
    elif param_fit["Em_gal_ana"]["use"]:
        model=  bkg + 1e-8*psf_SgrA(exposure*EmGal)+psf_SgrA(1e-8*exposure*mygaus_SgrA) + psf_G0p9(1e-8*exposure*mygaus_G0p9) 
    else:
        model=  bkg + psf_SgrA(1e-8*exposure*mygaus_SgrA) + psf_G0p9(1e-8*exposure*mygaus_G0p9) 
    set_full_model(model)
    map_data=SkyImage.empty_like(on)
    model_map =SkyImage.empty_like(on)
    resid =SkyImage.empty_like(on)
    map_data.data = get_data().y.reshape(shape) * mask
    model_map.data = get_model()(get_data().x0, get_data().x1).reshape(shape) * mask
    resid.data = map_data.data - model_map.data
    resid.write(outdir_result + "/residual_flux_fond_G0p9_SgrA_finalstep_"+ name + "_"  + str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.fits", clobber=True)
    map_data=SkyImage.empty_like(on)
    model_map =SkyImage.empty_like(on)
    resid =SkyImage.empty_like(on)
    map_data.data = get_data().y.reshape(shape) 
    model_map.data = get_model()(get_data().x0, get_data().x1).reshape(shape) 
    resid.data = map_data.data - model_map.data
    resid.write(outdir_result + "/residual_flux_fond_G0p9_SgrA_finalstep_whole_image"+ name + "_"  + str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.fits", clobber=True)
    table_models = table_models.filled(-1000)
    table_covar = table_covar.filled(0)
    filename_table_result = outdir_result + "/morphology_et_flux_fit_result_" + name + "_" + str("%.2f" % E1) + "_" + str(
        "%.2f" % E2) + "_TeV.txt"
    table_models.write(filename_table_result, format="ascii")
    filename_covar_result = outdir_result + "/morphology_et_flux_fit_covar_" + name + "_" + str("%.2f" % E1) + "_" + str(
        "%.2f" % E2) + "_TeV.txt"
    table_covar.write(filename_covar_result, format="ascii")
    table_models = Table()
    table_covar = Table()

    model=  bkg + psf_SgrA(1e-8*exposure*mygaus_SgrA) + psf_G0p9(1e-8*exposure*mygaus_G0p9)
    set_full_model(model)
    model_map =SkyImage.empty_like(on)
    model_map.data = get_model()(get_data().x0, get_data().x1).reshape(shape) * mask
    ncounts_sources_pontuelles=model_map.data.sum()
    map_data=SkyImage.empty_like(on)
    map_data.data = get_data().y.reshape(shape) * mask
    ntot=map_data.data.sum()
    print("Ntot= ", ntot)
    print("Ntot source ponctuelle ", ncounts_sources_pontuelles)
    if param_fit["Em_gal_hgps"]["use"]:
        model=  psf_SgrA(exposure*EmGal)
    if param_fit["Em_gal_ana"]["use"]:
        model=  1e-8*psf_SgrA(exposure*EmGal)
        set_full_model(model)
        model_map =SkyImage.empty_like(on)
        model_map.data = get_model()(get_data().x0, get_data().x1).reshape(shape) * mask
        ncounts_Em_gal=model_map.data.sum()
        print("Ntot Emgal ", ncounts_Em_gal)
        print("Ncoup_diffus/NtotEm_gal ", ncounts_Em_gal/(ntot-ncounts_sources_pontuelles))
        
    
"""    
color=["b","g","r", "m", "c"]
imin=0
imax=3
pt.figure()
for i in range(imin,imax):
    residus_l[i][np.where(np.isnan(residus_l[i]))]=0
    norm=residus_l[i].sum()
    pt.fill_between(l_rebin,residus_l[i]/norm+residus_err_l[i]/norm,residus_l[i]/norm-residus_err_l[i]/norm,color=color[i],alpha=0.3)
    #pt.errorbar(l_rebin, residus_l[i]/norm, yerr=residus_err_l[i]/norm,linestyle='None', marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV")
    pt.scatter(l_rebin, residus_l[i]/norm, marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV", color=color[i])
    pt.plot(l_rebin, residus_l[i]/norm, color=color[i])
    pt.legend(fontsize = 'x-small')
    pt.ylabel("residual")
    pt.xlabel("longitude (degrees)")
    pt.title("Residuals")
    pt.xlim(-1.5, 1.5)
    pt.gca().invert_xaxis()
    pt.savefig(outdir_profiles+"/profiles_longitude_superpostionresidual_finalstep_"+name+"_3premieresbandes_"  + str("%.2f" % energy_bins[imin].value) + "_" + str("%.2f" % energy_bins[imax].value) + "_TeV.png")

pt.figure()
for i in range(imin,imax):
    residus_b[i][np.where(np.isinf(residus_b[i]))]=np.nan
    residus_b[i][np.where(np.isnan(residus_b[i]))]=0
    residus_err_b[i][np.where(np.isinf(residus_err_b[i]))]=np.nan
    norm=residus_b[i].sum()
    pt.fill_between(b_rebin,residus_b[i]/norm+residus_err_b[i]/norm,residus_b[i]/norm-residus_err_b[i]/norm,color=color[i],alpha=0.3)
    pt.scatter(b_rebin, residus_b[i]/norm, marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV", color=color[i])
    pt.plot(b_rebin, residus_b[i]/norm, color=color[i])
    #pt.errorbar(b_rebin, residus_b[i]/norm, yerr=residus_err_b[i]/norm,linestyle='None', marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV")
    pt.legend(fontsize = 'x-small')
    pt.ylabel("residual")
    pt.xlabel("latitude (degrees)")
    pt.title("Residuals")
    pt.xlim(-1, 1)
    pt.savefig(outdir_profiles+"/profiles_lattitude_superpositionresidual_finalstep_"+name+"_3premieresbandes_"  + str("%.2f" % energy_bins[imin].value) + "_" + str("%.2f" % energy_bins[imax].value) + "_TeV.png")

imin=2
imax=4
pt.figure()
for i in range(imin,imax):
    residus_l[i][np.where(np.isnan(residus_l[i]))]=0
    norm=residus_l[i].sum()
    pt.fill_between(l_rebin,residus_l[i]/norm+residus_err_l[i]/norm,residus_l[i]/norm-residus_err_l[i]/norm,color=color[i],alpha=0.3)
    #pt.errorbar(l_rebin, residus_l[i]/norm, yerr=residus_err_l[i]/norm,linestyle='None', marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV")
    pt.scatter(l_rebin, residus_l[i]/norm, marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV", color=color[i])
    pt.plot(l_rebin, residus_l[i]/norm, color=color[i])
    pt.legend(fontsize = 'x-small')
    pt.ylabel("residual")
    pt.xlabel("longitude (degrees)")
    pt.title("Residuals")
    pt.xlim(-1.5, 1.5)
    pt.gca().invert_xaxis()
    pt.savefig(outdir_profiles+"/profiles_longitude_superpostionresidual_finalstep_"+name+"_2bandes_"  + str("%.2f" % energy_bins[imin].value) + "_" + str("%.2f" % energy_bins[imax].value) + "_TeV.png")

pt.figure()
for i in range(imin,imax):
    residus_b[i][np.where(np.isinf(residus_b[i]))]=np.nan
    residus_b[i][np.where(np.isnan(residus_b[i]))]=0
    residus_err_b[i][np.where(np.isinf(residus_err_b[i]))]=np.nan
    norm=residus_b[i].sum()
    pt.fill_between(b_rebin,residus_b[i]/norm+residus_err_b[i]/norm,residus_b[i]/norm-residus_err_b[i]/norm,color=color[i],alpha=0.3)
    pt.scatter(b_rebin, residus_b[i]/norm, marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV", color=color[i])
    pt.plot(b_rebin, residus_b[i]/norm, color=color[i])
    #pt.errorbar(b_rebin, residus_b[i]/norm, yerr=residus_err_b[i]/norm,linestyle='None', marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV")
    pt.legend(fontsize = 'x-small')
    pt.ylabel("residual")
    pt.xlabel("latitude (degrees)")
    pt.title("Residuals")
    pt.xlim(-1, 1)
    pt.savefig(outdir_profiles+"/profiles_lattitude_superpositionresidual_finalstep_"+name+"_2bandes_"  + str("%.2f" % energy_bins[imin].value) + "_" + str("%.2f" % energy_bins[imax].value) + "_TeV.png")

imin=0
imax=2
pt.figure()
for i in range(imin,imax):
    residus_l[i][np.where(np.isnan(residus_l[i]))]=0
    norm=residus_l[i].sum()
    pt.fill_between(l_rebin,residus_l[i]/norm+residus_err_l[i]/norm,residus_l[i]/norm-residus_err_l[i]/norm,color=color[i],alpha=0.3)
    #pt.errorbar(l_rebin, residus_l[i]/norm, yerr=residus_err_l[i]/norm,linestyle='None', marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV")
    pt.scatter(l_rebin, residus_l[i]/norm, marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV", color=color[i])
    pt.plot(l_rebin, residus_l[i]/norm, color=color[i])
    pt.legend(fontsize = 'x-small')
    pt.ylabel("residual")
    pt.xlabel("longitude (degrees)")
    pt.title("Residuals")
    pt.xlim(-1.5, 1.5)
    pt.gca().invert_xaxis()
    pt.savefig(outdir_profiles+"/profiles_longitude_superpostionresidual_finalstep_"+name+"_2bandes_"  + str("%.2f" % energy_bins[imin].value) + "_" + str("%.2f" % energy_bins[imax].value) + "_TeV.png")

imin=0
imax=2    
pt.figure()
for i in range(imin,imax):
    residus_b[i][np.where(np.isinf(residus_b[i]))]=np.nan
    residus_b[i][np.where(np.isnan(residus_b[i]))]=0
    residus_err_b[i][np.where(np.isinf(residus_err_b[i]))]=np.nan
    norm=residus_b[i].sum()
    pt.fill_between(b_rebin,residus_b[i]/norm+residus_err_b[i]/norm,residus_b[i]/norm-residus_err_b[i]/norm,color=color[i],alpha=0.3)
    pt.scatter(b_rebin, residus_b[i]/norm, marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV", color=color[i])
    pt.plot(b_rebin, residus_b[i]/norm, color=color[i])
    #pt.errorbar(b_rebin, residus_b[i]/norm, yerr=residus_err_b[i]/norm,linestyle='None', marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV")
    pt.legend(fontsize = 'x-small')
    pt.ylabel("residual")
    pt.xlabel("latitude (degrees)")
    pt.title("Residuals")
    pt.xlim(-1, 1)
    pt.savefig(outdir_profiles+"/profiles_lattitude_superpositionresidual_finalstep_"+name+"_2bandes_"  + str("%.2f" % energy_bins[imin].value) + "_" + str("%.2f" % energy_bins[imax].value) + "_TeV.png")


imin=0
imax=3
pt.figure()
for i in range(imin,imax):
    residus_l_SgrA_G0p9_arc_source[i][np.where(np.isnan(residus_l_SgrA_G0p9_arc_source[i]))]=0
    norm=residus_l_SgrA_G0p9_arc_source[i].sum()
    pt.fill_between(l_rebin,residus_l_SgrA_G0p9_arc_source[i]/norm+residus_err_l[i]/norm,residus_l_SgrA_G0p9_arc_source[i]/norm-residus_err_l[i]/norm,color=color[i],alpha=0.3)
    pt.scatter(l_rebin, residus_l_SgrA_G0p9_arc_source[i]/norm, marker="o",label="E:"+str("%.2f" % energy_bins[i].value)+" - "+str("%.2f" %energy_bins[i+1].value)+" TeV", color=color[i])
    pt.plot(l_rebin, residus_l_SgrA_G0p9_arc_source[i]/norm, color=color[i])
    pt.legend(fontsize = 'x-small')
    pt.ylabel("residual")
    pt.xlabel("longitude (degrees)")
    pt.title("Residuals")
    pt.xlim(-1.5, 1.5)
    pt.gca().invert_xaxis()
    pt.savefig(outdir_profiles+"/profiles_longitude_superpostionresidual_finalstep_avec_arcsource_"+name+"_3premieresbandes_"  + str("%.2f" % energy_bins[imin].value) + "_" + str("%.2f" % energy_bins[imax].value) + "_TeV.png")
"""
