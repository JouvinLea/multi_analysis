"""Define Sherpa models we use for the survey"""
import numpy as np
import sherpa.astro.ui as sau
from sherpa.astro.ui import erf
from sherpa.models import ArithmeticModel, Parameter

fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
fwhm_to_sigma_erf = np.sqrt(2) * fwhm_to_sigma
import json
        

# The implementation is quite easy: All models inherit from ArithmeticModel
# Parameters have to be set and self.calc has to be defined


class disk2d(ArithmeticModel):
    def __init__(self, name='disk2d'):
        self.xpos = Parameter(name, 'xpos', 0)  # p[0]
        self.ypos = Parameter(name, 'ypos', 0)  # p[1]
        self.ampl = Parameter(name, 'ampl', 1)  # p[2]
        self.r0 = Parameter(name, 'r0', 1, 0)  # p[3]
        ArithmeticModel.__init__(self, name, (self.xpos, self.ypos, self.ampl, self.r0))

    def calc(self, p, x, y, *args, **kwargs):
        # Compute radii
        r2 = (x - p[0]) ** 2 + (y - p[1]) ** 2

        # Return ampl when r2 <= r0 else return 0
        return np.select([r2 <= p[3] ** 2], [p[2]])

sau.add_model(disk2d)


class shell2d(ArithmeticModel):
    def __init__(self, name='shell2d'):
        self.xpos = Parameter(name, 'xpos', 0)  # p[0]
        self.ypos = Parameter(name, 'ypos', 0)  # p[1]
        self.ampl = Parameter(name, 'ampl', 1)  # p[2]
        self.r0 = Parameter(name, 'r0', 1, 0)  # p[3]
        self.width = Parameter(name, 'width', 0.1, 0)
        ArithmeticModel.__init__(self, name, (self.xpos, self.ypos, self.ampl, self.r0, self.width))

    def calc(self, p, x, y, *args, **kwargs):
        """Homogeneously emitting spherical shell,
        projected along the z-direction
        (this is not 100% correct for very large shells on the sky)."""
        (xpos, ypos, amplitude, r_in, width) = p
        rr = (x - xpos) ** 2 + (y - ypos) ** 2
        rr_in = r_in ** 2
        rr_out = (r_in + width) ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        # Note: for r > r_out 'np.select' fills automatically zeros!
        with np.errstate(invalid='ignore'):
            values = np.select([rr <= rr_in, rr <= rr_out],
                           [np.sqrt(rr_out - rr) - np.sqrt(rr_in - rr),
                            np.sqrt(rr_out - rr)])
        return amplitude * values / (2 * np.pi / 3 *
                                     (rr_out * (r_in + width) - rr_in * r_in))


sau.add_model(shell2d)


def normgauss2d_erf(p, x, y):
    """Evaluate 2d gaussian using the error function"""
    #import IPython; IPython.embed()
    sigma_erf = p[3] * fwhm_to_sigma_erf
    return p[2] / 4. * ((erf.calc.calc([1, p[0], sigma_erf], x + 0.5)
                     - erf.calc.calc([1, p[0], sigma_erf], x - 0.5))
                     * (erf.calc.calc([1, p[1], sigma_erf], y + 0.5)
                     - erf.calc.calc([1, p[1], sigma_erf], y - 0.5)))


class normgauss2dint(ArithmeticModel):
    def __init__(self, name='normgauss2dint'):
        # Gauss source parameters
        self.xpos = Parameter(name, 'xpos', 0)  # p[0]
        self.ypos = Parameter(name, 'ypos', 0)  # p[1]
        self.ampl = Parameter(name, 'ampl', 1)  # p[2]
        self.fwhm = Parameter(name, 'fwhm', 1)  # p[3]
        ArithmeticModel.__init__(self, name, (self.xpos, self.ypos, self.ampl, self.fwhm))

    def calc(self, p, x, y, *args, **kwargs):
        """
        The normgauss2dint model uses the error function to evaluate the
        the gaussian. This corresponds to an integration over bins.
        """
        return normgauss2d_erf(p, x, y)

sau.add_model(normgauss2dint)


class delta2dint(ArithmeticModel):
    def __init__(self, name='delta2dint'):
        # Gauss source parameters
        self.xpos = Parameter(name, 'xpos', 0)  # p[0]
        self.ypos = Parameter(name, 'ypos', 0)  # p[1]
        self.ampl = Parameter(name, 'ampl', 1)  # p[2]
        self.shape = sau.get_data().shape
        ArithmeticModel.__init__(self, name, (self.xpos, self.ypos, self.ampl))

    def calc(self, p, x, y, *args, **kwargs):
        """
        Evaluate using pixel fractions.
        """
        x_0_sub, x_0_pix = np.modf(p[0] - 1)
        y_0_sub, y_0_pix = np.modf(p[1] - 1)
        out = np.zeros(self.shape)
        out[y_0_pix, x_0_pix] = (1 - x_0_sub) * (1 - y_0_sub)
        out[y_0_pix + 1, x_0_pix + 1] = x_0_sub * y_0_sub
        out[y_0_pix + 1, x_0_pix] = (1 - x_0_sub) * y_0_sub
        out[y_0_pix, x_0_pix + 1] = x_0_sub * (1 - y_0_sub)
        return p[2] * out.flatten() 

sau.add_model(delta2dint)


c = 4 * np.log(2)

class hesssource2d(ArithmeticModel):
    def __init__(self, name='hesssource2d'):
        # Gauss source parameters
        self.xpos = Parameter(name, 'xpos', 0)  # p[0]
        self.ypos = Parameter(name, 'ypos', 0)  # p[1]
        self.ampl = Parameter(name, 'ampl', 1)  # p[2]
        self.fwhm = Parameter(name, 'fwhm', 1)  # p[3]
        self.loadpsf('psf.json')
        ArithmeticModel.__init__(self, name, (self.xpos, self.ypos, self.ampl, self.fwhm))
        
    def loadpsf(self, filename):
        self.psf = json.load(open(filename))
        
        integral = 0
        # Compute normalized amplitudes
        for psf in ['psf1', 'psf2', 'psf3']:
            self.psf[psf]['normampl'] = (self.psf[psf]['ampl'] * np.pi 
                                         / c * self.psf[psf]['fwhm'] ** 2)
            integral += self.psf[psf]['normampl']
        
        # Normalize psf integral to unity
        for psf in ['psf1', 'psf2', 'psf3']:
            self.psf[psf]['normampl'] /= integral
        
    def calc(self, p, x, y, *args, **kwargs):
        """Evaluate psf convolved Gaussian"""
        rr = (x - p[0]) ** 2 + (y - p[1]) ** 2
        ffwhm_1 = p[3] ** 2 + self.psf['psf1']['fwhm'] ** 2
        ffwhm_2 = p[3] ** 2 + self.psf['psf2']['fwhm'] ** 2
        ffwhm_3 = p[3] ** 2 + self.psf['psf3']['fwhm'] ** 2
        gauss_1 = self.psf['psf1']['normampl'] / ffwhm_1 * np.exp(- c * rr / ffwhm_1)
        gauss_2 = self.psf['psf2']['normampl'] / ffwhm_2 * np.exp(- c * rr / ffwhm_2)
        gauss_3 = self.psf['psf3']['normampl'] / ffwhm_3 * np.exp(- c * rr / ffwhm_3)
        return c * p[2] / np.pi * (gauss_1 + gauss_2 + gauss_3)

sau.add_model(hesssource2d)
