import numpy as np
from kapteyn import maputils,wcsgrat
from matplotlib import pyplot as plt
#from pylab import figure, axes, pie, title, show
from matplotlib.backends.backend_pdf import PdfPages
from kapteyn.mplutil import VariableColormap
import yaml
import sys
from gammapy.utils.energy import EnergyBounds, Energy
#from align_image import *
#from ds9_contours import *
from method_fit import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



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
if param_fit["central_gauss"]["fwhm_frozen"]:
    name += "_fwhmCC_"+str(param_fit["central_gauss"]["fwhm_init"])                                       
       

#fig = plt.figure(figsize=(14,8))
#fig = plt.figure(figsize=(14,24))
fig = plt.figure(figsize=(17,23))
frame_fig1 = fig.add_subplot(6,2,1)
frame_fig2 = fig.add_subplot(6,2,2)
frame_fig3 = fig.add_subplot(6,2,3)
frame_fig4 = fig.add_subplot(6,2,4)
frame_fig5 = fig.add_subplot(6,2,5)
frame_fig6 = fig.add_subplot(6,2,6)
frame_fig7 = fig.add_subplot(6,2,7)
frame_fig8 = fig.add_subplot(6,2,8)
frame_fig9 = fig.add_subplot(6,2,9)
frame_fig10 = fig.add_subplot(6,2,10)
frame_fig11 = fig.add_subplot(6,2,11)
frame_fig12 = fig.add_subplot(6,2,12)


resid_min = [-1.5, -1,-1, -1]
resid_max = [4.0,3.0,2.0,2.0]
resxmin = 25
resxmax = 225
resymin = 84
resymax = 164

smooth=2.

mod_min=-0.
mod_max=1.0
modxmin = 307
modxmax = 705
modymin = 413
modymax = 573

#E1=0.50
#E2=1.44
#E1=1.44
#E2=4.16
#E1=4.16
#E2=12.01
config_name = input_param["general"]["config_name"]
energy_reco=[Energy(input_param["energy binning"]["Emin"],"TeV"),Energy(input_param["energy binning"]["Emax"],"TeV"), input_param["energy binning"]["nbin"]]
energy_bins=EnergyBounds.equal_log_spacing(0.5,100,1,"TeV")
for i_E, E in enumerate(energy_bins):
    E1=energy_bins[i_E].value
    E2=energy_bins[i_E+1].value
    visible_grid = True
    delta_grid = 1.0
    minor_tick = 5
    palette = VariableColormap('../b.lut',name = 'b')
    #palette.set_scale('SQRT')
    directory=make_outdir_filesresult(source_name, name_method_fond,config_name,image_size,for_integral_flux=False,ereco=energy_reco)
    fileb = directory+'/residual_morpho_et_flux_step_'
    file_end = '_'+name+'_'+str("%.2f" % E1)+'_'+str("%.2f" % E2)+'_TeV.fits'
    filen = list()
    for i in range(7):
        filen.append(fileb + str(i+1)+ file_end)
        print filen[i],i

    gc = maputils.FITSimage("../Final_Plots_paper_Janv2016_LARGEFOV_StandardFit/Model_ipe_south_GAMMAEXP_HardZeta_PtSources_G09_CG_WCS.fits")
    #../Final_Plots_paper_Janv2016_LARGEFOV_StandardFit/Model_ipe_south_GAMMAEXP_HardZeta_bkg_Diff_CGPtSource_WCS.fits")
    gc.set_limits(pxlim=(modxmin,modxmax),pylim=(modymin,modymax))
    imgc = gc.Annotatedimage(frame_fig1,cmap=palette,clipmin=mod_min,clipmax=mod_max)
    imgc.Image(interpolation='spline36')
    grat_gc = imgc.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_gc.setp_gratline(visible=visible_grid)
    grat_gc.setp_tick(wcsaxis=[0,1],fontsize=14)
    grat_gc.setp_ticklabel(wcsaxis=[0],visible=False)
    grat_gc.setp_tickmark(color='w',markeredgewidth=3)
    grat_gc.setp_axislabel(plotaxis="bottom",visible=False)
    grat_gc.setp_axislabel(plotaxis="left",label="Galactic latitude",fontsize=14)
    #minor_gc = imgc.Minortickmarks(grat_gc, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)

    imgc.plot()


    gc1 = maputils.FITSimage(filen[0])
    gc1.set_limits(pxlim=(resxmin,resxmax),pylim=(resymin,resymax))
    img1 = gc1.Annotatedimage(frame_fig2,cmap=palette,clipmin=resid_min[i_E],clipmax=resid_max[i_E])
    img1.Image(interpolation='spline36')
    img1.set_blur(True, smooth, smooth, new=True)
    grat_g1 = img1.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_g1.setp_gratline(visible=visible_grid)
    grat_g1.setp_tick(wcsaxis=[0,1],fontsize=14)
    grat_g1.setp_ticklabel(wcsaxis=[0,1],visible=False)
    grat_g1.setp_tickmark(color='w',markeredgewidth=3)
    grat_g1.setp_axislabel(plotaxis="left",visible=False)
    grat_g1.setp_axislabel(plotaxis="bottom",visible=False)
    #minor_g1 = img1.Minortickmarks(grat_g1, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)
    img1.Marker(pos="ga 0.14 ga -0.114", marker='+', markersize=14, markeredgewidth=2, color='c')
    img1.Marker(pos="ga 0.66 ga -0.03", marker='+', markersize=14, markeredgewidth=2, color='c')

    img1.plot()

    gc2 = maputils.FITSimage("../Final_Plots_paper_Janv2016_LARGEFOV_StandardFit/Model_ipe_south_GAMMAEXP_HardZeta_CSGrad_WCS.fits")
    gc2.set_limits(pxlim=(modxmin,modxmax),pylim=(modymin,modymax))
    img2 = gc2.Annotatedimage(frame_fig3,cmap=palette,clipmin=mod_min,clipmax=mod_max)
    img2.Image(interpolation='spline36')
    grat_g2 = img2.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_g2.setp_gratline(visible=visible_grid)
    grat_g2.setp_tick(wcsaxis=[0,1],fontsize=14)
    grat_g2.setp_ticklabel(wcsaxis=[0],visible=False)
    grat_g2.setp_tickmark(color='w',markeredgewidth=3)
    grat_g2.setp_axislabel(plotaxis="bottom",visible=False)
    grat_g2.setp_axislabel(plotaxis="left",label="Galactic latitude",fontsize=14)
    #minor_g2 = img2.Minortickmarks(grat_g2, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)

    img2.plot()

    gc3 = maputils.FITSimage(filen[1])
    gc3.set_limits(pxlim=(resxmin,resxmax),pylim=(resymin,resymax))
    img3 = gc3.Annotatedimage(frame_fig4,cmap=palette,clipmin=resid_min[i_E],clipmax=resid_max[i_E])
    img3.Image(interpolation='spline36')
    img3.set_blur(True, smooth, smooth, new=True)
    grat_g3 = img3.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_g3.setp_gratline(visible=visible_grid)
    grat_g3.setp_tick(wcsaxis=[0,1],fontsize=14)
    grat_g3.setp_ticklabel(wcsaxis=[0,1],visible=False)
    grat_g3.setp_tickmark(color='w',markeredgewidth=3)
    grat_g3.setp_axislabel(plotaxis="left",visible=False)
    grat_g3.setp_axislabel(plotaxis="bottom",visible=False)
    #minor_g3 = img3.Minortickmarks(grat_g3, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)
    img3.Marker(pos="ga 0.14 ga -0.114", marker='+', markersize=14, markeredgewidth=2, color='c')
    img3.Marker(pos="ga 0.66 ga -0.03", marker='+', markersize=14, markeredgewidth=2, color='c')

    img3.plot()

    gc4 = maputils.FITSimage("../Final_Plots_paper_Janv2016_LARGEFOV_StandardFit/Model_ipe_south_GAMMAEXP_HardZeta_LargeScale_WCS.fits")
    gc4.set_limits(pxlim=(modxmin,modxmax),pylim=(modymin,modymax))
    img4 = gc4.Annotatedimage(frame_fig5,cmap=palette,clipmin=mod_min,clipmax=mod_max)
    img4.Image(interpolation='spline36')
    grat_g4 = img4.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_g4.setp_gratline(visible=visible_grid)
    grat_g4.setp_tick(wcsaxis=[0,1],fontsize=14)
    grat_g4.setp_ticklabel(wcsaxis=[0],visible=False)
    grat_g4.setp_tickmark(color='w',markeredgewidth=3)
    #grat_g4.setp_axislabel(plotaxis="bottom",label="Galactic longitude",fontsize=16)
    #grat_g4.setp_axislabel(plotaxis="left",visible=False)
    grat_g4.setp_axislabel(plotaxis="bottom",visible=False)
    grat_g4.setp_axislabel(plotaxis="left",label="Galactic latitude",fontsize=14)
    #minor_g4 = img4.Minortickmarks(grat_g4, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)

    #colbars = imgcs.Colorbar(fontsize=14)

    img4.plot()

    gc5 = maputils.FITSimage(filen[2])
    gc5.set_limits(pxlim=(resxmin,resxmax),pylim=(resymin,resymax))
    img5 = gc5.Annotatedimage(frame_fig6,cmap=palette,clipmin=resid_min[i_E],clipmax=resid_max[i_E])
    img5.Image(interpolation='spline36')
    img5.set_blur(True, smooth, smooth, new=True)
    grat_g5 = img5.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_g5.setp_gratline(visible=visible_grid)
    grat_g5.setp_tick(wcsaxis=[0,1],fontsize=14)
    grat_g5.setp_ticklabel(wcsaxis=[0,1],visible=False)
    grat_g5.setp_tickmark(color='w',markeredgewidth=3)
    #grat_g5.setp_axislabel(plotaxis="bottom",label="Galactic longitude",fontsize=12)
    grat_g5.setp_axislabel(plotaxis="bottom",visible=False)
    grat_g5.setp_axislabel(plotaxis="left",visible=False)
    #minor_g5 = img5.Minortickmarks(grat_g5, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)
    img5.Marker(pos="ga 0.14 ga -0.114", marker='+', markersize=14, markeredgewidth=2, color='c')
    img5.Marker(pos="ga 0.66 ga -0.03", marker='+', markersize=14, markeredgewidth=2, color='c')


    #grat_g5.setp_axislabel(plotaxis="left",label="Galactic latitude",fontsize=12)
    #colbars = imgcs.Colorbar(fontsize=14)
    img5.plot()

    gc6 = maputils.FITSimage("../Final_Plots_paper_Janv2016_LARGEFOV_StandardFit/Model_ipe_south_GAMMAEXP_HardZeta_CGBisGauss_WCS.fits")
    gc6.set_limits(pxlim=(modxmin,modxmax),pylim=(modymin,modymax))
    img6 = gc6.Annotatedimage(frame_fig7,cmap=palette,clipmin=mod_min,clipmax=mod_max)
    img6.Image(interpolation='spline36')
    grat_g6 = img6.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_g6.setp_gratline(visible=visible_grid)
    grat_g6.setp_tick(wcsaxis=[0,1],fontsize=14)
    grat_g6.setp_ticklabel(wcsaxis=[0],visible=False)
    grat_g6.setp_tickmark(color='w',markeredgewidth=3)
    #grat_g6.setp_axislabel(plotaxis="bottom",label="Galactic longitude",fontsize=16)
    #grat_g6.setp_axislabel(plotaxis="left",visible=False)
    grat_g6.setp_axislabel(plotaxis="bottom",visible=False)
    grat_g6.setp_axislabel(plotaxis="left",label="Galactic latitude",fontsize=14)
    #colbars = imgcs.Colorbar(fontsize=14)
    #minor_g6 = img6.Minortickmarks(grat_g6, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)


    img6.plot()


    gc7 = maputils.FITSimage(filen[3])
    gc7.set_limits(pxlim=(resxmin,resxmax),pylim=(resymin,resymax))
    img7 = gc7.Annotatedimage(frame_fig8,cmap=palette,clipmin=resid_min[i_E],clipmax=resid_max[i_E])
    img7.Image(interpolation='spline36')
    img7.set_blur(True, smooth, smooth, new=True)
    grat_g7 = img7.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_g7.setp_gratline(visible=visible_grid)
    grat_g7.setp_tick(wcsaxis=[0,1],fontsize=14)
    grat_g7.setp_ticklabel(wcsaxis=[0,1],visible=False)
    grat_g7.setp_tickmark(color='w',markeredgewidth=3)
    #grat_g7.setp_axislabel(plotaxis="bottom",label="Galactic longitude",fontsize=12)
    grat_g7.setp_axislabel(plotaxis="left",visible=False)
    grat_g7.setp_axislabel(plotaxis="bottom",visible=False)
    #minor_g7 = img7.Minortickmarks(grat_g7, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)
    img7.Marker(pos="ga 0.14 ga -0.114", marker='+', markersize=14, markeredgewidth=2, color='c')
    img7.Marker(pos="ga 0.66 ga -0.03", marker='+', markersize=14, markeredgewidth=2, color='c')


    #grat_g7.setp_axislabel(plotaxis="left",label="Galactic latitude",fontsize=12)

    img7.plot()


    gc8 = maputils.FITSimage("../Final_Plots_paper_Janv2016_LARGEFOV_StandardFit/Model_ipe_south_GAMMAEXP_HardZeta_G0p13_PtSource_WCS.fits")
    gc8.set_limits(pxlim=(modxmin,modxmax),pylim=(modymin,modymax))
    img8 = gc8.Annotatedimage(frame_fig9,cmap=palette,clipmin=mod_min,clipmax=mod_max)
    img8.Image(interpolation='spline36')
    grat_g8 = img8.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_g8.setp_gratline(visible=visible_grid)
    grat_g8.setp_tick(wcsaxis=[0,1],fontsize=14)
    grat_g8.setp_ticklabel(wcsaxis=[0],visible=False)
    grat_g8.setp_tickmark(color='w',markeredgewidth=3)
    #grat_g8.setp_axislabel(plotaxis="bottom",visible=False)#label="Galactic longitude",fontsize=14)
    #grat_g8.setp_axislabel(plotaxis="left",visible=False)
    grat_g8.setp_axislabel(plotaxis="bottom",visible=False)
    grat_g8.setp_axislabel(plotaxis="left",label="Galactic latitude",fontsize=14)
    #minor_g8 = img8.Minortickmarks(grat_g8, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)


    img8.plot()

    gc9 = maputils.FITSimage(filen[4])
    gc9.set_limits(pxlim=(resxmin,resxmax),pylim=(resymin,resymax))
    img9 = gc9.Annotatedimage(frame_fig10,cmap=palette,clipmin=resid_min[i_E],clipmax=resid_max[i_E])
    img9.Image(interpolation='spline36')
    img9.set_blur(True, smooth, smooth, new=True)
    grat_g9 = img9.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_g9.setp_gratline(visible=visible_grid)
    grat_g9.setp_tick(wcsaxis=[0,1],fontsize=14)
    grat_g9.setp_ticklabel(wcsaxis=[1],visible=False)
    grat_g9.setp_tickmark(color='w',markeredgewidth=3)
    grat_g9.setp_axislabel(plotaxis="bottom",visible=False)#label="Galactic longitude",fontsize=14)
    grat_g9.setp_axislabel(plotaxis="left",visible=False)
    #minor_g9 = img9.Minortickmarks(grat_g9, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)
    img9.Marker(pos="ga 0.14 ga -0.114", marker='+', markersize=14, markeredgewidth=2, color='c')
    img9.Marker(pos="ga 0.66 ga -0.03", marker='+', markersize=14, markeredgewidth=2, color='c')

    #grat_g9.setp_axislabel(plotaxis="left",label="Galactic latitude",fontsize=14)
    img9.plot()


    gc10 = maputils.FITSimage("../Final_Plots_paper_Janv2016_LARGEFOV_StandardFit/Model_ipe_south_GAMMAEXP_HardZeta_SgrB2_PtSource_WCS.fits")
    gc10.set_limits(pxlim=(modxmin,modxmax),pylim=(modymin,modymax))
    img10 = gc10.Annotatedimage(frame_fig11,cmap=palette,clipmin=mod_min,clipmax=mod_max)
    img10.Image(interpolation='spline36')
    grat_g10 = img10.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_g10.setp_gratline(visible=visible_grid)
    grat_g10.setp_tick(wcsaxis=[0,1],fontsize=14)
    #grat_g8.setp_ticklabel(wcsaxis=[0],visible=False)
    grat_g10.setp_tickmark(color='w',markeredgewidth=3)
    grat_g10.setp_axislabel(plotaxis="bottom",label="Galactic longitude",fontsize=14)
    #grat_g8.setp_axislabel(plotaxis="left",visible=False)
    #grat_g8.setp_axislabel(plotaxis="bottom",visible=False)
    grat_g10.setp_axislabel(plotaxis="left",label="Galactic latitude",fontsize=14)
    #minor_g8 = img8.Minortickmarks(grat_g8, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)


    img10.plot()

    gc11 = maputils.FITSimage(filen[5])
    gc11.set_limits(pxlim=(resxmin,resxmax),pylim=(resymin,resymax))
    img11 = gc11.Annotatedimage(frame_fig12,cmap=palette,clipmin=resid_min[i_E],clipmax=resid_max[i_E])
    img11.Image(interpolation='spline36')
    img11.set_blur(True, smooth, smooth, new=True)
    grat_g11 = img11.Graticule(unitsx='DEGREE',unitsy='DEGREE',deltax=delta_grid,deltay=delta_grid)
    grat_g11.setp_gratline(visible=visible_grid)
    grat_g11.setp_tick(wcsaxis=[0,1],fontsize=14)
    grat_g11.setp_ticklabel(wcsaxis=[1],visible=False)
    grat_g11.setp_tickmark(color='w',markeredgewidth=3)
    grat_g11.setp_axislabel(plotaxis="bottom",label="Galactic longitude",fontsize=14)
    grat_g11.setp_axislabel(plotaxis="left",visible=False)
    #minor_g9 = img9.Minortickmarks(grat_g9, minor_tick, minor_tick, color="w", markersize=3, markeredgewidth=2)
    img11.Marker(pos="ga 0.14 ga -0.114", marker='+', markersize=14, markeredgewidth=2, color='c')
    img11.Marker(pos="ga 0.66 ga -0.03", marker='+', markersize=14, markeredgewidth=2, color='c')

    #grat_g9.setp_axislabel(plotaxis="left",label="Galactic latitude",fontsize=14)
    img11.plot()

    #fig.subplots_adjust(hspace=-0.155,wspace=0)
    fig.subplots_adjust(hspace=-0.405,wspace=0.)

    #maputils.showall()



    #fig.savefig('file_iterative_b.png',bbox_inches='tight')
    fig.savefig('plot_CG/file_iterative_b'+name+'_'+str("%.2f"%E1)+'_'+str("%.2f"%E2)+'_TeV.png')
    plt.close(fig)
