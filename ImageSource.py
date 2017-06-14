import numpy as np
from astropy.table import Table
import astropy.units as u
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from astropy.units import Quantity
from astropy.table import Column
from gammapy.datasets import gammapy_extra
from gammapy.background import EnergyOffsetBackgroundModel
from gammapy.utils.energy import EnergyBounds, Energy
from gammapy.data import DataStore
from gammapy.utils.axis import sqrt_space
from gammapy.image import SkyImage, SkyImageList
from gammapy.background import fill_acceptance_image
from gammapy.stats import significance
from gammapy.background import OffDataBackgroundMaker
from gammapy.data import ObservationTable
import matplotlib.pyplot as plt
from gammapy.utils.scripts import make_path
from gammapy.extern.pathlib import Path
from gammapy.scripts import StackedObsImageMaker
from gammapy.data import ObservationList
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from gammapy.data import ObservationTable
from gammapy.utils.scripts import make_path
from gammapy.extern.pathlib import Path
from gammapy.scripts import StackedObsImageMaker
from gammapy.data import ObservationList
from gammapy.image import SkyImage
from gammapy.data import DataStore
from gammapy.spectrum import LogEnergyAxis
from gammapy.cube import SkyCube, StackedObsCubeMaker
from gammapy.irf import TablePSF
from gammapy.utils.fits import energy_axis_to_ebounds
from  gammapy.utils.nddata import BinnedDataAxis
import os
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

import shutil

"""
Method to compute the images and psf of a given source
"""


def make_outdir(source_name, name_bkg,config_name,image_size,for_integral_flux, ereco,etrue=None,use_cube=False, use_etrue=False):
    """

    Parameters
    ----------
    source_name: name of the source you want to compute the image
    name_bkg: name of the bkg model you use to produce your bkg image
    config_name: 
    image_size:
    for_integral_flux: True if you want to compute the exposure to get the integral flux
    ereco: Tuple for the energy reco bin: (Emin,Emax,nbins)
    etrue: Tuple for the energy true bin: (Emin,Emax,nbins)
    use_cube: True if you want to compute cube analisys
    use_etrue: True if you want to compute the exposure cube and psf mean cube in true energy
    

    Returns
    -------
    directory where your fits file will go
    """
    n_binE=ereco[2]
    emin_reco=ereco[0].value
    emax_reco=ereco[1].value
    outdir = os.path.expandvars('$Image') +"/"+config_name + "/Image_" + source_name + "_bkg_" + name_bkg + "/binE_" + str(n_binE) +"_min_"+str(emin_reco)+"_max_"+str(emax_reco)+"_size_image_"+str(image_size)+"_pix"
    if not for_integral_flux:
        outdir+= "_exposure_flux_diff"
    if use_cube:
        outdir+= "_cube_images"
        if use_etrue:
            n_binE_true=etrue[2]
            emin_true=etrue[0].value
            emax_true=etrue[1].value
            outdir+= "_use_etrue_min_"+str(emin_true)+"_max_"+str(emax_true)+"_bin_"+str(n_binE_true)
    try:
        shutil.rmtree(outdir)
    except Exception:
        pass
    make_path(outdir).mkdir()
    return outdir


def make_obsdir(source_name, name_bkg,config_name):
    """

    Parameters
    ----------
    source_name: name of the source you want to compute the image
    name_bkg: name of the bkg model you use to produce your bkg image

    Returns
    -------
    directory where your data for your image are located
    """
    return os.path.expandvars('$Image') +"/" +config_name + "/Image_" + source_name + "_bkg_" + name_bkg + "/data"


def make_new_directorydataset(nobs, config_directory, source_name, center, obsdir, use_taget_name=False,source_name_skycoord=None):
    """
    Creates a directory with only the run used for the Images of the source and create a new index table with the
    background aceeptance curve location to used for the bkg image

    Parameters
    ----------
    nobs: number of observation you want
    config_directory: name of the config chains used to produce the data
    source_name: name of the source you want to compute the image
    center: SkyCoord of the source
    obsdir: directory where you want to put these data
    use_taget_name: True if you want to use only the observation that have this source for Target
    source_name_skycoord: if use_taget_name on true, name of the source target in the ds.obs_table
    Returns
    -------

    """
    ds = DataStore.from_dir(config_directory)
    obs = ds.obs_table
    # center = SkyCoord.from_name("Crab")
    pointing = SkyCoord(obs["RA_PNT"], obs["DEC_PNT"], unit='deg', frame='fk5')
    sep = center.separation(pointing)
    i = np.where(sep < 2 * u.deg)
    obs_table_target = obs[i]
    if use_taget_name:
        itarget=np.where(obs_table_target["OBJECT"]==source_name_skycoord)[0]
        obs_table_target=obs_table_target[itarget]
    if "QUALITY" in obs_table_target.colnames:
        ds.obs_table = obs_table_target[obs_table_target["QUALITY"] == 0]
    else:
        ds.obs_table = obs_table_target
    try:
        shutil.rmtree(obsdir)
    except Exception:
        pass
    ds.copy_obs(ds.obs_table[0:nobs], obsdir)


def make_new_directorydataset_listobs(nobs, config_directory, source_name, center, obsdir, list_obs):
    """
    Creates a directory with only the run used for the Images of the source and create a new index table with the
    background aceeptance curve location to used for the bkg image.
    Used a list of observation not a selection at less tat 2deg from the pointing position

    Parameters
    ----------
    nobs: number of observation you want
    config_directory: name of the config chains used to produce the data
    source_name: name of the source you want to compute the image
    center: SkyCoord of the source
    obsdir: directory where you want to put these data
    list_obs: list of obs id we want to create the new data_store
    Returns
    -------

    """
    ds = DataStore.from_dir(config_directory)
    list_ind = list()
    for obs in list_obs:
        try:
            i = np.where(ds.obs_table["OBS_ID"] == obs)[0][0]
            list_ind.append(i)
        except Exception:
            print(obs)
            continue
    
    try:
        shutil.rmtree(obsdir)
    except Exception:
        pass
    ds.copy_obs(ds.obs_table[list_ind], obsdir)


def add_bkgmodel_to_indextable(bkg_model_directory, source_name, obsdir):
    """
    Creates an indextable with the location of the bkg files you want to use to compute the bkg model

    Parameters
    ----------
    bkg_model_directory: directory where is located the bkg model you want to use for your bkg image
    source_name: name of the source you want to compute the image
    obsdir: directory where you want to put these data

    Returns
    -------

    """
    ds = DataStore.from_dir(obsdir)
    bgmaker = OffDataBackgroundMaker(ds)
    bkg_model_outdir = Path(bkg_model_directory)
    group_filename = str(bkg_model_outdir / 'group_def.fits')
    index_table = bgmaker.make_total_index_table(ds, "2D", bkg_model_outdir, group_filename, True)
    fn = obsdir + '/hdu-index.fits.gz'
    index_table.write(fn, overwrite=True)


def make_images(image_size, energy_band, offset_band, center, data_store, obs_table_subset, exclusion_mask, outdir,
                make_background_image=True, spectral_index=2.3, for_integral_flux=False, radius=10.,save_bkg_norm=True):
    """
    MAke the counts, bkg, mask, exposure, significance and ecxees images
    Parameters
    ----------
    energy_band: energy band on which you want to compute the map
    offset_band: offset band on which you want to compute the map
    center: SkyCoord of the source
    data_store: DataStore object containing the data used to coompute the image
    obs_table_subset: obs_table of the data_store containing the observations you want to use to compute the image. Could
    be smaller than the one of the datastore
    exclusion_mask: SkyMask used for the escluded regions
    outdir: directory where the fits image will go
    make_background_image: if you want to compute the bkg for the images. Most of the case yes otherwise there is only the counts image
    spectral_index: assumed spectral index to compute the exposure
    for_integral_flux: True if you want to get the inegrak flux with the exposure
    radius: Disk radius in pixels for the significance

    Returns
    -------

    """
    # TODO: fix `binarize` implementation
    image = SkyImage.empty(nxpix=image_size, nypix=image_size, binsz=0.02, xref=center.galactic.l.deg,
                           yref=center.galactic.b.deg, proj='TAN', coordsys='GAL')
    refheader = image.to_image_hdu().header
    exclusion_mask = exclusion_mask.reproject(reference=refheader)
    mosaicimages = StackedObsImageMaker(image, energy_band=energy_band, offset_band=offset_band, data_store=data_store,
                                        obs_table=obs_table_subset, exclusion_mask=exclusion_mask,save_bkg_scale=save_bkg_norm)
    mosaicimages.make_images(make_background_image=make_background_image, spectral_index=spectral_index,
                             for_integral_flux=for_integral_flux, radius=radius)
    filename = outdir + '/fov_bg_maps' + str(energy_band[0].value) + '_' + str(energy_band[1].value) + '_TeV.fits'
    if 'COMMENT' in mosaicimages.images["exclusion"].meta:
        del mosaicimages.images["exclusion"].meta['COMMENT']   
    write_mosaic_images(mosaicimages, filename)
    return mosaicimages

def make_images_several_energyband(image_size,energy_bins, offset_band, source_name, center, data_store, obs_table_subset,
                                   exclusion_mask, outdir, make_background_image=True, spectral_index=2.3,
                                   for_integral_flux=False, radius=10.,save_bkg_norm=True):
    """
    MAke the counts, bkg, mask, exposure, significance and ecxees images for different energy bands

    Parameters
    ----------
    energy_bins: array of energy bands on which you want to compute the map
    offset_band: offset band on which you want to compute the map
    center: SkyCoord of the source
    data_store: DataStore object containing the data used to coompute the image
    obs_table_subset: obs_table of the data_store containing the observations you want to use to compute the image. Could
    be smaller than the one of the datastore
    exclusion_mask: SkyMask used for the escluded regions
    outdir: directory where the fits image will go
    make_background_image: if you want to compute the bkg for the images. Most of the case yes otherwise there is only the counts image
    spectral_index: assumed spectral index to compute the exposure
    for_integral_flux: True if you want to get the inegrak flux with the exposure
    radius: Disk radius in pixels for the significance

    Returns
    -------
    """
    list_mosaicimages=list()
    for i, E in enumerate(energy_bins[0:-1]):
        energy_band = Energy([energy_bins[i].value, energy_bins[i + 1].value], energy_bins.unit)
        print energy_band
        mosaicimages=make_images(image_size,energy_band, offset_band, center, data_store, obs_table_subset, exclusion_mask, outdir,
                    make_background_image, spectral_index, for_integral_flux, radius,save_bkg_norm)
        list_mosaicimages.append(mosaicimages)
    if save_bkg_norm:
        table=Table()
        for i,mosaic_images in enumerate(list_mosaicimages):
            table_bkg=mosaic_images.table_bkg_scale
            if i==0:
                array_bkg_scale=np.zeros((len(obs_table_subset),len(energy_bins[0:-1])))
                array_counts=np.zeros((len(obs_table_subset),len(energy_bins[0:-1])))
            itot=0    
            for irun,run in enumerate(table_bkg["OBS_ID"]):
                while run!=obs_table_subset["OBS_ID"][itot]:
                    itot+=1
                array_bkg_scale[itot,i]=table_bkg["bkg_scale"][irun]
                array_counts[itot,i]=table_bkg["N_counts"][irun]
                itot+=1
        c0 = fits.Column(name="OBS_ID", format='E', array=table_bkg["OBS_ID"].data)
        c1 = fits.Column(name="bkg_norm", format='PE()', array=array_bkg_scale)
        c2 = fits.Column(name="counts", format='PE()', array=array_counts)
        hdu = fits.BinTableHDU.from_columns([c0, c1,c2])
        ebounds = energy_axis_to_ebounds(energy_bins)
        #ebounds = energy_axis_to_ebounds(BinnedDataAxis(energy_bins[0:-1],energy_bins[1:]))
        prim_hdu = fits.PrimaryHDU()
        hdu_list=fits.HDUList([prim_hdu, hdu, ebounds])
        hdu_list.writeto(outdir + "/table_bkg_norm_.fits") 
                
            

def make_empty_cube(image_size, energy,center, data_unit=""):
    """
    Parameters
    ----------
    image_size:int, Total number of pixel of the 2D map
    energy: Tuple for the energy axis: (Emin,Emax,nbins)
    center: SkyCoord of the source
    unit : str, Data unit.
    """
    def_image=dict()
    def_image["nxpix"]=image_size
    def_image["nypix"]=image_size
    def_image["binsz"]=0.02
    def_image["xref"]=center.galactic.l.deg
    def_image["yref"]=center.galactic.b.deg
    def_image["proj"]='TAN'
    def_image["coordsys"]='GAL'
    def_image["unit"]=data_unit
    e_min, e_max, nbins=energy
    empty_cube=SkyCube.empty(emin=e_min.value, emax=e_max.value, enumbins=nbins, eunit=e_min.unit, mode='edges', **def_image)
    return empty_cube
    
    

def make_cube(image_size, energy_reco, energy_true, offset_band, center, data_store, obs_table_subset, exclusion_mask, outdir,
                make_background_image=True, radius=10.,save_bkg_norm=True):
    """
    MAke the counts, bkg, mask, exposure, significance and excess images

    Parameters
    ----------
    image_size:int, Total number of pixel of the 2D map
    energy_reco: Tuple for the energy reco bin: (Emin,Emax,nbins)
    energy_true: Tuple for the energy true bin: (Emin,Emax,nbins)
    offset_band: offset band on which you want to compute the map
    center: SkyCoord of the source
    data_store: DataStore object containing the data used to coompute the image
    obs_table_subset: obs_table of the data_store containing the observations you want to use to compute the image. Could
    be smaller than the one of the datastore
    exclusion_mask: SkyMask used for the excluded regions
    outdir: directory where the fits image will go
    make_background_image: if you want to compute the bkg for the images. Most of the case yes otherwise there is only the counts image
    radius: Disk radius in pixels for the significance

    Returns
    -------

    """
    # TODO: fix `binarize` implementation
    #ref_cube_images=make_empty_cube(image_size, energy_reco,center, data_unit="ct")
    ref_cube_images=make_empty_cube(image_size, energy_reco,center)
    ref_cube_exposure=make_empty_cube(image_size, energy_true,center, data_unit="m2 s")
    ref_cube_skymask=make_empty_cube(image_size, energy_reco,center)
    
    refheader = ref_cube_images.sky_image_ref.to_image_hdu().header
    exclusion_mask = exclusion_mask.reproject(reference=refheader)
    ref_cube_skymask.data=np.tile(exclusion_mask.data,(energy_reco[2], 1, 1))
    
    #mosaic_cubes = StackedObsCubeMaker(empty_cube_images=ref_cube_images, empty_exposure_cube=ref_cube_exposure, offset_band=offset_band, data_store=data_store, obs_table=obs_table_subset, exclusion_mask=exclusion_mask,save_bkg_scale=save_bkg_norm)
    mosaic_cubes = StackedObsCubeMaker(empty_cube_images=ref_cube_images, empty_exposure_cube=ref_cube_exposure, offset_band=offset_band, data_store=data_store, obs_table=obs_table_subset, exclusion_mask=ref_cube_skymask,save_bkg_scale=save_bkg_norm)
    mosaic_cubes.make_cubes(make_background_image=make_background_image, radius=radius)
    ctot=np.sum(mosaic_cubes.counts_cube.data * ref_cube_skymask.data,axis=(1,2))
    btot=np.sum(mosaic_cubes.bkg_cube.data * ref_cube_skymask.data,axis=(1,2))
    scale=ctot/btot
    for iE,E in enumerate(mosaic_cubes.bkg_cube.energies()):
        mosaic_cubes.bkg_cube.data[iE,:,:]*=scale[iE]
    if 'COMMENT' in exclusion_mask.meta:
        del exclusion_mask.meta['COMMENT']
    filename_mask=outdir + '/exclusion_mask.fits'
    filename_counts = outdir + '/counts_cube.fits'
    filename_bkg = outdir + '/bkg_cube.fits'
    filename_significance = outdir + '/significance_cube.fits'
    filename_excess = outdir + '/excess_cube.fits'
    filename_exposure = outdir + '/exposure_cube.fits'
    exclusion_mask.write(filename_mask, clobber=True)
    mosaic_cubes.counts_cube.write(filename_counts,format="fermi-counts")
    mosaic_cubes.bkg_cube.write(filename_bkg,format="fermi-counts")
    mosaic_cubes.significance_cube.write(filename_significance,format="fermi-counts")
    mosaic_cubes.excess_cube.write(filename_excess,format="fermi-counts")
    #mosaic_cubes.exposure_cube.write(filename_exposure,format="fermi-exposure")
    mosaic_cubes.exposure_cube.write(filename_exposure,format="fermi-counts")
    if save_bkg_norm:
        filename_bkg_norm = outdir + '/table_bkg_norm_.fits'
        mosaic_cubes.table_bkg_scale.write(filename_bkg_norm,overwrite=True)



def make_psf(energy_band, source_name, center, ObsList, outdir, spectral_index=2.3):
    """
    Compute the mean psf for a set of observation and a given energy band
    Parameters
    ----------
    energy_band: energy band on which you want to compute the map
    source_name: name of the source you want to compute the image
    center: SkyCoord of the source
    ObsList: ObservationList to use to compute the psf (could be different that the data_store for G0p9 for the GC for example)
    outdir: directory where the fits image will go
    spectral_index: assumed spectral index to compute the psf

    Returns
    -------

    """   
    energy = EnergyBounds.equal_log_spacing(energy_band[0].value, energy_band[1].value, 100, energy_band.unit)
    # Here all the observations have a center at less than 2 degrees from the Crab so it will be ok to estimate the mean psf on the Crab source postion (the area is define for offset equal to 2 degrees...)
    psf_energydependent = ObsList.make_mean_psf(center, energy, theta=None)
    #import IPython; IPython.embed()
    try:
        psf_table = psf_energydependent.table_psf_in_energy_band(energy_band, spectral_index=spectral_index)
    except:
        psf_table=TablePSF(psf_energydependent.offset, Quantity(np.zeros(len(psf_energydependent.offset)),u.sr**-1))
    Table_psf = Table()
    c1 = Column(psf_table._dp_domega, name='psf_value', unit=psf_table._dp_domega.unit)
    c2 = Column(psf_table._offset, name='theta', unit=psf_table._offset.unit)
    Table_psf.add_column(c1)
    Table_psf.add_column(c2)
    filename_psf = outdir + "/psf_table_" + source_name + "_" + str(energy_band[0].value) + '_' + str(
        energy_band[1].value) + ".fits"
    Table_psf.write(filename_psf, overwrite=True)
    

def make_psf_several_energyband(energy_bins, source_name, center, ObsList, outdir,
                                spectral_index=2.3):
    """
    Compute the mean psf for a set of observation for different energy bands
    Parameters
    ----------
    energy_band: energy band on which you want to compute the map
    source_name: name of the source you want to compute the image
    center: SkyCoord of the source
    ObsList: ObservationList to use to compute the psf (could be different that the data_store for G0p9 for the GC for example)
    outdir: directory where the fits image will go
    spectral_index: assumed spectral index to compute the psf

    Returns
    -------

    """
    for i, E in enumerate(energy_bins[0:-1]):
        energy_band = Energy([energy_bins[i].value, energy_bins[i + 1].value], energy_bins.unit)
        print energy_band
        make_psf(energy_band, source_name, center, ObsList, outdir, spectral_index)

def make_psf_cube(image_size,energy_cube, source_name, center_maps, center, ObsList, outdir,
                                spectral_index=2.3):
    """
    Compute the mean psf for a set of observation for different energy bands
    Parameters
    ----------
    image_size:int, Total number of pixel of the 2D map
    energy: Tuple for the energy axis: (Emin,Emax,nbins)
    source_name: name of the source you want to compute the image
    center_maps: SkyCoord
            center of the images
    center: SkyCoord 
            position where we want to compute the psf
    ObsList: ObservationList to use to compute the psf (could be different that the data_store for G0p9 for the GC for example)
    outdir: directory where the fits image will go
    spectral_index: assumed spectral index to compute the psf

    Returns
    -------

    """
    ref_cube=make_empty_cube(image_size, energy_cube,center_maps)
    header = ref_cube.sky_image_ref.to_image_hdu().header
    energy_bins=ref_cube.energies(mode="edges")
    for i_E, E in enumerate(energy_bins[0:-1]):
        energy_band = Energy([energy_bins[i_E].value, energy_bins[i_E + 1].value], energy_bins.unit)
        energy = EnergyBounds.equal_log_spacing(energy_band[0].value, energy_band[1].value, 100, energy_band.unit)
        # Here all the observations have a center at less than 2 degrees from the Crab so it will be ok to estimate the mean psf on the Crab source postion (the area is define for offset equal to 2 degrees...)
        psf_energydependent = ObsList.make_mean_psf(center, energy, theta=None)
        try:
            psf_table = psf_energydependent.table_psf_in_energy_band(energy_band, spectral_index=spectral_index)
        except:
            psf_table=TablePSF(psf_energydependent.offset, Quantity(np.zeros(len(psf_energydependent.offset)),u.sr**-1))    
        ref_cube.data[i_E,:,:] = fill_acceptance_image(header, center_maps, psf_table._offset.to("deg") ,psf_table._dp_domega, psf_table._offset.to("deg")[-1]).data 
    ref_cube.write(outdir+"/mean_psf_cube_"+source_name+".fits", format="fermi-counts")
        

def make_mean_rmf(energy_true, energy_reco, center, ObsList, outdir, source_name=""):
    """
    Compute the mean psf for a set of observation and a given energy band
    Parameters
    ----------
    energy_true: Tuple for the energy true bin: (Emin,Emax,nbins)
    energy_reco: Tuple for the energy reco bin: (Emin,Emax,nbins)
    center: SkyCoord of the source
    ObsList: ObservationList to use to compute the psf (could be different that the data_store for G0p9 for the GC for example)
    outdir: directory where the fits image will go
    source_name: name of the source for which you want to compute the mean RMF


    Returns
    -------

    """
    
    # Here all the observations have a center at less than 2 degrees from the Crab so it will be ok to estimate the mean psf on the Crab source postion (the area is define for offset equal to 2 degrees...)
    emin_true, emax_true, nbin_true = energy_true
    emin_reco, emax_reco, nbin_reco = energy_reco
    energy_true_bins = EnergyBounds.equal_log_spacing(emin_true, emax_true, nbin_true, 'TeV')
    energy_reco_bins = EnergyBounds.equal_log_spacing(emin_reco, emax_reco, nbin_reco, 'TeV')
    rmf = ObsList.make_mean_edisp(position=center,e_true= energy_true_bins, e_reco=energy_reco_bins)
    rmf.write(outdir+"/mean_rmf"+source_name+".fits", clobber=True)
    

        
def write_mosaic_images(mosaicimages, filename):
    """
    Write mosaicimages
    Parameters
    ----------
    mosaicimages
    filename

    Returns
    -------

    """
    log.info('Writing {}'.format(filename))
    mosaicimages.images.write(filename, clobber=True)
