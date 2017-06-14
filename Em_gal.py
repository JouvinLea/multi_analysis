import numpy as np
from sherpa.models import ArithmeticModel, Parameter
import astropy.wcs as pywcs

class UniformGaussianPlane(ArithmeticModel):
   """ Uniform Gaussian band model 
   """

   def __init__(self,wcs,name='UniformGaussianPlane'):
       """ Default initialization 
       """
       self.__name__ = name
       self.wcs = pywcs.WCS()
       self.binsize = 1.0
       self.coordsys = "galactic"  # default
       #self.ampl = Parameter(name,'ampl',1.0,0.0,1e6)
       #self.thick = Parameter(name,'thick',0.4,0.,1.0)
       #self.ypos = Parameter(name,'ypos',0.0,-1.5,1.5)
       """
       self.ampl = Parameter(name,'ampl',1.0,0.0,1e6)
       self.thick = Parameter(name,'thick',15,0.,50)
       self.ypos = Parameter(name,'ypos',200,125,275)
       """
       #import IPython; IPython.embed()
       self.binsize_deg = np.abs(wcs.wcs.cdelt[0])
       res = wcs.wcs_world2pix(np.array([0.,0],ndmin=2),1)
       xpix,ypix = res[0]
       #self.binsize_deg=0.02
       #ypix=202.81000059
       self.ampl = Parameter(name,'ampl',1.0, 0.0,1e6)
       self.thick = Parameter(name,'thick',0.3/self.binsize_deg,0.,1/self.binsize_deg)
       self.ypos = Parameter(name,'ypos',ypix,ypix-(1.5/self.binsize_deg),ypix+(1.5/self.binsize_deg))
       
       
       ArithmeticModel.__init__(self,name,(self.ypos,self.thick,self.ampl))

   def set_parameters(self, ypos,thick,ampl,err_ypos=0.2,thick_max=1.5):
       """ Define source parameters
       """
       self.thick.set(val=thick,min=0.0,max=thick_max)
       self.ypos.set(val=ypos,min=-err_ypos,max=err_ypos)
       self.ampl.set(val=ampl)

   def set_wcs(self, wcs):

      self.wcs = wcs
      # We assume bins have the same size along x and y axis
      self.binsize = np.abs(self.wcs.wcs.cdelt[0])
      if self.wcs.wcs.ctype[0][0:4] == 'GLON':
         self.coordsys = 'galactic'
      elif self.wcs.wcs.ctype[0][0:2] == 'RA':
         self.coordsys = 'fk5'

   def calc(self, pars,x,y, *args, **kwargs):
       (ypos,thick, ampl) = pars
       res = self.wcs.wcs_world2pix(np.array([0.,ypos],ndmin=2),1)
       xpix,ypix = res[0]
       thr = thick/self.binsize
       d2 = (y-ypix)**2
       tsq = thr*thr
       return ampl*np.exp(-0.5*d2/tsq)
