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
                                        param_fit["gauss_SgrA"]["fwhm_frozen"], None,
                                        param_fit["gauss_SgrA"]["ampl_frozen"], xpos_SgrA,
                                        param_fit["gauss_SgrA"]["xpos_frozen"],
                                        ypos_SgrA, param_fit["gauss_SgrA"]["ypos_frozen"])
    # modele gauss pour G0p9 centre sur G0p9
    source_center_G0p9 = SkyCoord(input_param["param_G0p9"]["sourde_name_skycoord"]["l_gal"],
                                  input_param["param_G0p9"]["sourde_name_skycoord"]["b_gal"], unit='deg',
                                  frame="galactic")
    xpos_G0p9, ypos_G0p9 = skycoord_to_pixel(source_center_G0p9, on.wcs)
    xpos_G0p9 += 0.5
    ypos_G0p9 += 0.5
    mygaus_G0p9 = source_punctual_model(param_fit["gauss_G0p9"]["name"], param_fit["gauss_G0p9"]["fwhm_init"],
                                        param_fit["gauss_G0p9"]["fwhm_frozen"], None,
                                        param_fit["gauss_G0p9"]["ampl_frozen"], xpos_G0p9,
                                        param_fit["gauss_G0p9"]["xpos_frozen"],
                                        ypos_G0p9, param_fit["gauss_G0p9"]["ypos_frozen"])

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
    central_gauss = source_punctual_model(param_fit["central_gauss"]["name"], None,
                                param_fit["central_gauss"]["fwhm_frozen"], None,
                                param_fit["central_gauss"]["ampl_frozen"], xpos_GC,
                                param_fit["central_gauss"]["xpos_frozen"],
                                ypos_GC, param_fit["central_gauss"]["ypos_frozen"])
    #Arc_source
    source_center_arcsource = SkyCoord(param_fit["arc source"]["l"],
                       param_fit["arc source"]["b"], unit='deg', frame="galactic")
    xpos_arcsource, ypos_arcsource = skycoord_to_pixel(source_center_arcsource, on.wcs)
    arc_source=source_punctual_model(param_fit["arc source"]["name"], param_fit["arc source"]["fwhm_init"],
                                       param_fit["arc source"]["fwhm_frozen"], None,
                                       param_fit["arc source"]["ampl_frozen"], xpos_arcsource, param_fit["arc source"]["xpos_frozen"],
                          ypos_arcsource, param_fit["arc source"]["ypos_frozen"])
    #Gauss SgrB2
    source_center_sgrB2 = SkyCoord(param_fit["SgrB2"]["l"],
                       param_fit["SgrB2"]["b"], unit='deg', frame="galactic")
    xpos_sgrB2, ypos_sgrB2 = skycoord_to_pixel(source_center_sgrB2, on.wcs)
    sgrB2=source_punctual_model(param_fit["SgrB2"]["name"], param_fit["SgrB2"]["fwhm_init"],
                                       param_fit["SgrB2"]["fwhm_frozen"], None,
                                       param_fit["SgrB2"]["ampl_frozen"], xpos_sgrB2, param_fit["SgrB2"]["xpos_frozen"],
                          ypos_sgrB2, param_fit["SgrB2"]["ypos_frozen"])
    """
    # Modele symetric central gauss centre sur SgrA
    central_gauss = source_NormGauss2D(param_fit["central_gauss"]["name"], None,
                                param_fit["central_gauss"]["fwhm_frozen"], None,
                                param_fit["central_gauss"]["ampl_frozen"], xpos_GC,
                                param_fit["central_gauss"]["xpos_frozen"],
                                ypos_GC, param_fit["central_gauss"]["ypos_frozen"])
    #Arc_source
    source_center_arcsource = SkyCoord(param_fit["arc source"]["l"],
                       param_fit["arc source"]["b"], unit='deg', frame="galactic")
    xpos_arcsource, ypos_arcsource = skycoord_to_pixel(source_center_arcsource, on.wcs)
    arc_source=source_NormGauss2D(param_fit["arc source"]["name"], param_fit["arc source"]["fwhm_init"],
                                       param_fit["arc source"]["fwhm_frozen"], None,
                                       param_fit["arc source"]["ampl_frozen"], xpos_arcsource, param_fit["arc source"]["xpos_frozen"],
                          ypos_arcsource, param_fit["arc source"]["ypos_frozen"])
    #Gauss SgrB2
    source_center_sgrB2 = SkyCoord(param_fit["SgrB2"]["l"],
                       param_fit["SgrB2"]["b"], unit='deg', frame="galactic")
    xpos_sgrB2, ypos_sgrB2 = skycoord_to_pixel(source_center_sgrB2, on.wcs)
    sgrB2=source_NormGauss2D(param_fit["SgrB2"]["name"], param_fit["SgrB2"]["fwhm_init"],
                                       param_fit["SgrB2"]["fwhm_frozen"], None,
                                       param_fit["SgrB2"]["ampl_frozen"], xpos_sgrB2, param_fit["SgrB2"]["xpos_frozen"],
                          ypos_sgrB2, param_fit["SgrB2"]["ypos_frozen"])
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

    list_src = [psf_SgrA(mygaus_SgrA)]
    if param_fit["gauss_G0p9"]["fit"]:
        list_src.append(psf_G0p9(mygaus_G0p9))
    # Si on inverse LS et CS alors c est qu il y a les deux!
    if param_fit["invert_CS_LS"]:
        list_src.append(psf_SgrA(gaus_CS * CS))
        if param_fit["invert_CC_LS"]:
            list_src.append(psf_SgrA(central_gauss))    
            list_src.append(psf_SgrA(Large_Scale))
        else:    
            list_src.append(psf_SgrA(Large_Scale))
            list_src.append(psf_SgrA(central_gauss))
    else:
        if param_fit["Large scale"]["fit"]:
            list_src.append(psf_SgrA(Large_Scale))
        if param_fit["Gauss_to_CS"]["fit"]:
            list_src.append(psf_SgrA(gaus_CS * CS))
        if param_fit["central_gauss"]["fit"]:
            list_src.append(psf_SgrA(central_gauss))
    if param_fit["arc source"]["fit"]:
        list_src.append(psf_SgrA(arc_source))
    if param_fit["SgrB2"]["fit"]:
        list_src.append(psf_SgrA(sgrB2))

    model = bkg
    set_full_model(model)
    pdf_lat=PdfPages(outdir_profiles+"/profiles_lattitude_"+name+"_" + str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.pdf")
    pdf_lon=PdfPages(outdir_profiles+"/profiles_longitude_"+name+"_" + str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.pdf")
    for i_src, src in enumerate(list_src):
        model += src
        set_full_model(model)
        fit()
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
        shape = np.shape(on.data)
        mask = get_data().mask.reshape(shape)
        #import IPython; IPython.embed()
        #exposure=SkyImage.read(outdir_data+"/exp_maps0.5_100.0_TeV.fits")
        #exposure.data=exposure.data*mask
        #pt.figure(1)
        #exposure.plot(add_cbar=True)
        #pt.savefig("test_exposure.png")
        
        map_data=SkyImage.empty_like(on)
        model_map =SkyImage.empty_like(on)
        resid =SkyImage.empty_like(on)
        exp_map=SkyImage.empty_like(on)
        map_data.data = get_data().y.reshape(shape) * mask
        model_map.data = get_model()(get_data().x0, get_data().x1).reshape(shape) * mask
        exp_map.data= np.ones(map_data.data.shape)* mask
        
        resid.data = map_data.data - model_map.data
        resid.write(outdir_result + "/residual_morpho_step_" + str(i_src) + "_"+ name + "_"  + str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.fits", clobber=True)
        coord = on.coordinates()

        # Longitude profile
        i_b = np.where((coord.b[:, 0] < on.center.b + 0.15 * u.deg) & (coord.b[:, 0] > on.center.b - 0.15 * u.deg))[0]
        npix_l = np.sum(np.flipud(mask[i_b, :]), axis=0)
        l = coord.l[0, :]
        l.value[np.where(l > 180 * u.deg)] = l.value[np.where(l > 180 * u.deg)] - 360
        profile_l_on = np.sum(map_data.data[i_b, :], axis=0) / npix_l
        profile_l_model = np.sum(model_map.data[i_b, :], axis=0) / npix_l
        profile_l_resid = np.sum(resid.data[i_b, :], axis=0) / npix_l
        err_l = np.sqrt(profile_l_on / npix_l)
        nrebin_l=3
        l_rebin=rebin_profile2(l.value, nrebin_l)
        npix_l_rebin = rebin_profile2(npix_l, nrebin_l)
        profile_l_on_rebin=rebin_profile2(profile_l_on, nrebin_l)
        resid_l_rebin=rebin_profile2(profile_l_resid, nrebin_l)
        err_l_rebin = np.sqrt(profile_l_on_rebin / npix_l_rebin)
        # Ca donne des coups par arcmin2 car on prend en compte qu on ne cumula pas le meme nombre de pixel pour chaque
        # longitude vu qu il y a des regions d exclusions
        # Latitude profile
        l_center = on.center.l
        if l_center > 180 * u.deg:
            l_center = l_center - 360 * u.deg
        i_l = np.where((l < l_center + 1.5 * u.deg) & (l > l_center - 1.5 * u.deg))[0]
        npix_b = np.sum(np.flipud(mask[:, i_l]), axis=1)
        profile_b_on = np.sum(map_data.data[:, i_l], axis=1) / npix_b
        profile_b_model = np.sum(model_map.data[:, i_l], axis=1) / npix_b
        profile_b_resid = np.sum(resid.data[:, i_l], axis=1) / npix_b
        err_b = np.sqrt(profile_b_on / npix_b)
        nrebin_b=3
        b_rebin = rebin_profile2(coord.b[:, 0].value, nrebin_b)
        npix_b_rebin = rebin_profile2(npix_b, nrebin_b)
        profile_b_on_rebin=rebin_profile2(profile_b_on, nrebin_b)
        resid_b_rebin=rebin_profile2(profile_b_resid, nrebin_b)
        err_b_rebin = np.sqrt(profile_b_on_rebin / npix_b_rebin)

        fig = pt.figure()
        ax = fig.add_subplot(2, 1, 1)
        pt.plot(l.value, profile_l_model, label="model")
        pt.plot(l.value, profile_l_on, label="on data")
        pt.xlim(-1.5, 1.5)
        pt.gca().invert_xaxis()
        pt.legend()
        ax = fig.add_subplot(2, 1, 2)
        pt.errorbar(l_rebin, resid_l_rebin, yerr=err_l_rebin, linestyle='None', marker="o",
                    label="Step= " + str(i_src))
        pt.axhline(y=0, color='red', linewidth=2)
        pt.legend()
        pt.ylabel("residual")
        pt.xlabel("longitude (degrees)")
        pt.title("longitude profile")
        pt.xlim(-1.5, 1.5)
        pt.gca().invert_xaxis()
        pdf_lon.savefig()

        fig = pt.figure()
        ax = fig.add_subplot(2, 1, 1)
        pt.plot(coord.b[:, 0].value, profile_b_model, label="model")
        pt.plot(coord.b[:, 0].value, profile_b_on, label="on data")
        pt.xlim(-1, 1)
        pt.legend()
        ax = fig.add_subplot(2, 1, 2)
        pt.errorbar(b_rebin, resid_b_rebin, yerr=err_b_rebin, linestyle='None', marker="o",
                    label="Step= " + str(i_src))
        pt.axhline(y=0, color='red', linewidth=2)
        pt.legend()
        pt.ylabel("residual")
        pt.xlabel("latitude (degrees)")
        pt.title("latitude profile")
        pt.xlim(-1, 1)
        pdf_lat.savefig()

        E_center = EnergyBounds(energy_band).log_centers
        if E_center < 1 * u.TeV:
            pix = 5
        elif ((1 * u.TeV < E_center) & (E_center < 5 * u.TeV)):
            pix = 4
        else:
            pix = 2.5
        kernel = Gaussian2DKernel(pix)
        TS = compute_ts_image(map_data, model_map, exp_map, kernel)
        TS.write(outdir_plot+"/TS_map_step_" + str(i_src) + "_" +name+"_"+ str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.fits",
                 clobber=True)
        sig = SkyImage.empty(TS["ts"])
        sig.data = np.sqrt(TS["ts"].data)
        sig.name = "sig"
        sig.write(
            outdir_plot+"/significance_map_step_" + str(i_src) + "_" +name+"_" +str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.fits",
            clobber=True)
        if i_src==len(list_src)-1:
            # Profil lattitude et longitude
            shape = np.shape(on.data)
            mask = get_data().mask.reshape(shape)
            map_data=SkyImage.empty_like(on)
            model_map =SkyImage.empty_like(on)
            map_data.data = get_data().y.reshape(shape) * mask


            coord = on.coordinates()
            list_model=list()
            list_name_model=list()
            list_data_model=list()
            list_data_resid=list()
            if input_param["param_fit"]["use_EM_model"]:
            #on fixe la valeur de l'amplitude obtenu dans une autre region
                EmGal=make_EG_model(outdir_data, on,Emgal_evaluated,True)
                model=bkg + psf(EmGal) + psf_SgrA(mygaus_SgrA) + psf_G0p9(mygaus_G0p9)   
            else:
                model = bkg + psf_SgrA(mygaus_SgrA) + psf_G0p9(mygaus_G0p9)
            list_model.append(model)
            list_name_model.append("GC source + G0.9")
            if param_fit["invert_CS_LS"]:
                model+=psf_SgrA(gaus_CS * CS)
                list_model.append(model)
                model+=psf_SgrA(Large_Scale)
                list_model.append(model)
                list_name_model.append("Gauss*Templ_CS")
                list_name_model.append("Asym Large Scale")
            else:
                if param_fit["Large scale"]["fit"]:
                    model+=psf_SgrA(Large_Scale)
                    list_model.append(model)
                    list_name_model.append("Asym Large Scale")
                if param_fit["Gauss_to_CS"]["fit"]:
                    model+=psf_SgrA(gaus_CS * CS)
                    list_model.append(model)
                    list_name_model.append("Gauss*Templ_CS")
            if param_fit["central_gauss"]["fit"]:
                model+=psf_SgrA(central_gauss)
                list_model.append(model)
                list_name_model.append("Central Component")
            if param_fit["arc source"]["fit"]:
                model+=psf_SgrA(arc_source)
                list_model.append(model)
                list_name_model.append("Arc source")
            if param_fit["SgrB2"]["fit"]:
                model+=psf_SgrA(sgrB2)
                list_model.append(model)
                list_name_model.append("SgrB2")
            for model in list_model:
                set_full_model(model)
                model_map.data = get_model()(get_data().x0, get_data().x1).reshape(shape) * mask
                resid_maps = map_data.data - model_map.data
                list_data_model.append(model_map.data)
                list_data_resid.append(resid_maps)
            #only SgrA+G0P9+arc source model pour etre sur qu il ne reste pas l arc source dans les residus finaux
            model=  bkg + psf_SgrA(mygaus_SgrA) + psf_G0p9(mygaus_G0p9) + psf_SgrA(arc_source)
            set_full_model(model)
            model_map_SgrA_G0p9_arc_source_model = get_model()(get_data().x0, get_data().x1).reshape(shape) * mask
            resid_maps_SgrA_G0p9_arc_source_model = map_data.data - model_map_SgrA_G0p9_arc_source_model
            exp_map.data= np.ones(map_data.data.shape)*mask
            resid=SkyImage.empty_like(on)
            resid.data = map_data.data - list_data_model[0]
            resid.write(outdir_result + "/residual_morpho_finalstep_"+ name + "_"  + str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.fits", clobber=True)


            exposure_map=SkyImageList.read(outdir_data + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["exposure"].data
            # Longitude profile
            i_b = np.where((coord.b[:, 0] < on.center.b + 0.15 * u.deg) & (coord.b[:, 0] > on.center.b - 0.15 * u.deg))[0]
            npix_l = np.sum(np.flipud(mask[i_b, :]), axis=0)
            l = coord.l[0, :]
            l.value[np.where(l > 180 * u.deg)] = l.value[np.where(l > 180 * u.deg)] - 360
            profile_l_on = np.sum(map_data.data[i_b, :], axis=0) / npix_l
            profile_l_exposure = np.sum(exposure_map[i_b, :], axis=0) / npix_l
            err_l = np.sqrt(profile_l_on / npix_l)
            nrebin_l=3
            l_rebin=rebin_profile2(l.value, nrebin_l)
            npix_l_rebin = rebin_profile2(npix_l, nrebin_l)
            profile_l_on_rebin=rebin_profile2(profile_l_on, nrebin_l)
            err_l_rebin = np.sqrt(profile_l_on_rebin / npix_l_rebin)

            # Ca donne des coups par arcmin2 car on prend en compte qu on ne cumula pas le meme nombre de pixel pour chaque
            # longitude vu qu il y a des regions d exclusions


            # Latitude profile
            l_center = on.center.l
            if l_center > 180 * u.deg:
                l_center = l_center - 360 * u.deg
            i_l = np.where((l < l_center + 0.15 * u.deg) & (l > l_center - 0.15 * u.deg))[0]
            npix_b = np.sum(np.flipud(mask[:, i_l]), axis=1)
            profile_b_on = np.sum(map_data.data[:, i_l], axis=1) / npix_b
            profile_b_exposure = np.sum(exposure_map[:, i_l], axis=1) / npix_b
            err_b = np.sqrt(profile_b_on / npix_b)
            nrebin_b=3
            b_rebin = rebin_profile2(coord.b[:, 0].value, nrebin_b)
            npix_b_rebin = rebin_profile2(npix_b, nrebin_b)
            profile_b_on_rebin=rebin_profile2(profile_b_on, nrebin_b)
            err_b_rebin = np.sqrt(profile_b_on_rebin / npix_b_rebin)

            profile_l_models=np.zeros((len(npix_l),len(list_model)))
            profile_l_resids=np.zeros((len(npix_l),len(list_model)))
            resid_l_rebins=np.zeros((len(npix_l_rebin),len(list_model)))
            profile_b_models=np.zeros((len(npix_b),len(list_model)))
            profile_b_resids=np.zeros((len(npix_b),len(list_model)))
            resid_b_rebins=np.zeros((len(npix_b_rebin),len(list_model)))
            for i_model,model in enumerate(list_model):
                profile_l_models[:,i_model]=np.sum(list_data_model[i_model][i_b, :], axis=0) / npix_l
                profile_b_models[:,i_model]=np.sum(list_data_model[i_model][:, i_l], axis=1) / npix_b
                profile_l_resids[:,i_model] = np.sum(list_data_resid[i_model][i_b, :], axis=0) / npix_l
                resid_l_rebins[:,i_model] = rebin_profile2(profile_l_resids[:,i_model], nrebin=3)
                profile_b_resids[:,i_model] = np.sum(list_data_resid[i_model][:, i_l], axis=1) / npix_b
                resid_b_rebins[:,i_model] = rebin_profile2(profile_b_resids[:,i_model], nrebin=3)
            
            color=["b","r","g","c","m","grey","brown"]
            residus_l.append(resid_l_rebins[:,0])
            residus_b.append(resid_b_rebins[:,0])
            residus_err_l.append(err_l_rebin)
            residus_err_b.append(err_b_rebin)
            profile_l_SgrA_G0p9_arc_source=np.sum(model_map_SgrA_G0p9_arc_source_model[i_b, :], axis=0) / npix_l
            resid_l_SgrA_G0p9_arc_source=np.sum(resid_maps_SgrA_G0p9_arc_source_model[i_b, :], axis=0) / npix_l
            resid_l_rebin_SgrA_G0p9_arc_source=rebin_profile2(resid_l_SgrA_G0p9_arc_source, nrebin=3)
            residus_l_SgrA_G0p9_arc_source.append(resid_l_rebin_SgrA_G0p9_arc_source)
            for i_model,model in enumerate(list_model):
                fig = pt.figure()
                ax = fig.add_subplot(2, 1, 1)
                pt.plot(l.value, profile_l_on, label="on data")
                pt.plot(l.value, profile_l_models[:,i_model], color=color[1],label=list_name_model[i_model])
                pt.xlim(-1.5, 1.5)
                pt.gca().invert_xaxis()
                pt.legend()
                ax = fig.add_subplot(2, 1, 2)
                pt.errorbar(l_rebin, resid_l_rebins[:,i_model], yerr=err_l_rebin, color=color[1],linestyle='None', marker="o")
                pt.axhline(y=0, color='black',linewidth=2)
                pt.legend(fontsize = 'x-small')
                pt.ylabel("residual")
                pt.xlabel("longitude (degrees)")
                pt.title("")
                pt.xlim(-1.5, 1.5)
                pt.gca().invert_xaxis()
                pdf_lon.savefig()

            fig = pt.figure()
            ax = fig.add_subplot(2, 1, 1)
            pt.plot(l.value, profile_l_exposure, label="exposure")
            pt.xlim(-1.5, 1.5)
            pt.gca().invert_xaxis()
            pt.legend()
            pt.xlabel("longitude (degrees)")
            pt.title("exposure")
            pt.xlim(-1.5, 1.5)
            pt.gca().invert_xaxis()
            pdf_lon.savefig()

            fig = pt.figure()
            ax = fig.add_subplot(2, 1, 1)
            pt.plot(l.value, profile_l_on, label="on data")
            for i_model,model in enumerate(list_model):
                pt.plot(l.value, profile_l_models[:,i_model], color=color[i_model+1],label=list_name_model[i_model])
            pt.xlim(-1.5, 1.5)
            pt.gca().invert_xaxis()
            pt.legend()
            ax = fig.add_subplot(2, 1, 2)
            for i_model,model in enumerate(list_model):
                pt.errorbar(l_rebin, resid_l_rebins[:,i_model], yerr=err_l_rebin, color=color[i_model+1],linestyle='None', marker="o",label=list_name_model[i_model])
            pt.axhline(y=0, color='black',linewidth=2)
            pt.legend(fontsize = 'x-small')
            pt.ylabel("residual")
            pt.xlabel("longitude (degrees)")
            pt.title("Final step: different components")
            pt.xlim(-1.5, 1.5)
            pt.gca().invert_xaxis()
            pdf_lon.savefig()

            fig = pt.figure()
            ax = fig.add_subplot(2, 1, 1)
            pt.plot(l.value, profile_l_on, label="on data")
            for i_model,model in enumerate(list_model[0:-2]):
                pt.plot(l.value, profile_l_models[:,i_model], color=color[i_model+1],label=list_name_model[i_model])
            pt.xlim(-1.5, 1.5)
            pt.gca().invert_xaxis()
            pt.legend()
            ax = fig.add_subplot(2, 1, 2)
            for i_model,model in enumerate(list_model[0:-2]):
                pt.errorbar(l_rebin, resid_l_rebins[:,i_model], yerr=err_l_rebin, color=color[i_model+1],linestyle='None', marker="o",label=list_name_model[i_model])
            pt.axhline(y=0, color='black', linewidth=2)
            pt.legend(fontsize = 'x-small')
            pt.ylabel("residual")
            pt.xlabel("longitude (degrees)")
            pt.title("Final step: different components")
            pt.xlim(-1.5, 1.5)
            pt.gca().invert_xaxis()
            pdf_lon.savefig()


            fig = pt.figure()
            ax = fig.add_subplot(2, 1, 1)
            pt.plot(l.value, profile_l_on, label="on data")
            for i_model in range(3,6):
                pt.plot(l.value, profile_l_models[:,i_model], color=color[i_model+1],label=list_name_model[i_model])
            pt.xlim(-1.5, 1.5)
            pt.gca().invert_xaxis()
            pt.legend()
            ax = fig.add_subplot(2, 1, 2)
            for i_model in range(3,6):
                pt.errorbar(l_rebin, resid_l_rebins[:,i_model], yerr=err_l_rebin, color=color[i_model+1],linestyle='None', marker="o",label=list_name_model[i_model])
            pt.axhline(y=0, color='black', linewidth=2)
            pt.legend(fontsize = 'x-small')
            pt.ylabel("residual")
            pt.xlabel("longitude (degrees)")
            pt.title("Final step: different components")
            pt.xlim(-1.5, 1.5)
            pt.gca().invert_xaxis()
            pdf_lon.savefig()

            for i_model,model in enumerate(list_model):
                fig = pt.figure()
                ax = fig.add_subplot(2, 1, 1)
                pt.plot(coord.b[:, 0].value, profile_b_on, label="on data")
                pt.plot(coord.b[:, 0].value, profile_b_models[:,i_model], color=color[1],label=list_name_model[i_model])
                pt.xlim(-1, 1)
                pt.legend()
                ax = fig.add_subplot(2, 1, 2)
                pt.errorbar(b_rebin, resid_b_rebins[:,i_model], yerr=err_b_rebin, color=color[1],linestyle='None', marker="o")
                pt.axhline(y=0, color='black', linewidth=2)
                pt.legend(fontsize = 'x-small')
                pt.ylabel("residual")
                pt.xlabel("latitude (degrees)")
                #pt.title("latitude profile")
                pt.xlim(-1, 1)
                pdf_lat.savefig()

            fig = pt.figure()
            ax = fig.add_subplot(2, 1, 1)
            pt.plot(coord.b[:, 0].value, profile_b_exposure, label="exposure")
            pt.xlim(-1, 1)
            pt.legend()
            pt.xlabel("latitude (degrees)")
            pt.title("Exposure")
            pt.xlim(-1, 1)
            pdf_lat.savefig()

            fig = pt.figure()
            ax = fig.add_subplot(2, 1, 1)
            pt.plot(coord.b[:, 0].value, profile_b_on, label="on data")
            for i_model,model in enumerate(list_model):
                pt.plot(coord.b[:, 0].value, profile_b_models[:,i_model], color=color[i_model+1],label=list_name_model[i_model])
            pt.xlim(-1, 1)
            pt.legend()
            ax = fig.add_subplot(2, 1, 2)
            for i_model,model in enumerate(list_model):
                pt.errorbar(b_rebin, resid_b_rebins[:,i_model], yerr=err_b_rebin, color=color[i_model+1],linestyle='None', marker="o",label=list_name_model[i_model])
            pt.axhline(y=0, color='black', linewidth=2)
            pt.legend(fontsize = 'x-small')
            pt.ylabel("residual")
            pt.xlabel("latitude (degrees)")
            pt.title("latitude profile")
            pt.xlim(-1, 1)
            pdf_lat.savefig()

            fig = pt.figure()
            ax = fig.add_subplot(2, 1, 1)
            pt.plot(coord.b[:, 0].value, profile_b_on, label="on data")
            for i_model,model in enumerate(list_model[0:-2]):
                pt.plot(coord.b[:, 0].value, profile_b_models[:,i_model], color=color[i_model+1],label=list_name_model[i_model])
            pt.xlim(-1, 1)
            pt.legend()
            ax = fig.add_subplot(2, 1, 2)
            for i_model,model in enumerate(list_model[0:-2]):
                pt.errorbar(b_rebin, resid_b_rebins[:,i_model], yerr=err_b_rebin, color=color[i_model+1],linestyle='None', marker="o",label=list_name_model[i_model])
            pt.axhline(y=0, color='black', linewidth=2)
            pt.legend(fontsize = 'x-small')
            pt.ylabel("residual")
            pt.xlabel("latitude (degrees)")
            pt.title("latitude profile")
            pt.xlim(-1, 1)
            pdf_lat.savefig()

            fig = pt.figure()
            ax = fig.add_subplot(2, 1, 1)
            pt.plot(coord.b[:, 0].value, profile_b_on, label="on data")
            for i_model in range(3,6):
                pt.plot(coord.b[:, 0].value, profile_b_models[:,i_model], color=color[i_model+1],label=list_name_model[i_model])
            pt.xlim(-1, 1)
            pt.legend()
            ax = fig.add_subplot(2, 1, 2)
            for i_model in range(3,6):
                pt.errorbar(b_rebin, resid_b_rebins[:,i_model], yerr=err_b_rebin, color=color[i_model+1],linestyle='None', marker="o",label=list_name_model[i_model])
            pt.axhline(y=0, color='black', linewidth=2)
            pt.legend(fontsize = 'x-small')
            pt.ylabel("residual")
            pt.xlabel("latitude (degrees)")
            pt.title("latitude profile")
            pt.xlim(-1, 1)
            pdf_lat.savefig()

    pdf_lon.close()
    pdf_lat.close()
    table_models = table_models.filled(-1000)
    table_covar = table_covar.filled(0)
    filename_table_result = outdir_result + "/morphology_fit_result_" + name + "_" + str("%.2f" % E1) + "_" + str(
        "%.2f" % E2) + "_TeV.txt"
    table_models.write(filename_table_result, format="ascii")
    filename_covar_result = outdir_result + "/morphology_fit_covar_" + name + "_" + str("%.2f" % E1) + "_" + str(
        "%.2f" % E2) + "_TeV.txt"
    table_covar.write(filename_covar_result, format="ascii")
    table_models = Table()
    table_covar = Table()
    
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
