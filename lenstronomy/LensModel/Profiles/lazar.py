__author__ = 'sibirrer'

# this file contains a class to compute the Navaro-Frenk-White profile
import numpy as np
import lenstronomy.Util.util as util

import scipy.interpolate as interp
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil

__all__ = ['Lazar']

class Lazar(LensProfileBase, SersicUtil):
    """
    this class contains functions concerning the Lazar profile

    relation are: R_200 = c * Rs
    The definition of 'Rs' is in angular (arc second) units and the normalization is put in in regards to a deflection
    angle at 'Rs' - 'alpha_Rs'. To convert a physical mass and concentration definition into those lensing quantities
    for a specific redshift configuration and cosmological model, you can find routines in lenstronomy.Cosmo.lens_cosmo.py

    Examples for converting angular to physical mass units
    ------------------------------------------------------
    >>> from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    >>> lens_cosmo = LensCosmo(z_lens=0.5, z_source=1.5, cosmo=cosmo)

    Here we compute the angular scale of Rs on the sky (in arc seconds) and the deflection angle at Rs (in arc seconds):

    >>> Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=10**13, c=6)

    And here we perform the inverse calculation given Rs_angle and alpha_Rs to return the physical halo properties.

    >>> rho0, Rs, c, r200, M200 = lens_cosmo.nfw_angle2physical(Rs_angle=Rs_angle, alpha_Rs=alpha_Rs)

    The lens model calculation uses angular units as arguments! So to execute a deflection angle calculation one uses

    >>> from lenstronomy.LensModel.Profiles.nfw import NFW
    >>> nfw = NFW()
    >>> alpha_x, alpha_y = nfw.derivatives(x=1, y=1, Rs=Rs_angle, alpha_Rs=alpha_Rs, center_x=0, center_y=0)

    """

    profile_name = 'Lazar'

    ###### start modifying things here.
    param_names = ['k_eff', 'R_sersic', 'center_x', 'center_y']
    lower_limit_default = {'k_eff': 0, 'R_sersic': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'k_eff': 10, 'R_sersic': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        # the profile is the same form as the Sersics profile.
        # the mapping follows as n --> 1/beta, b --> 1/beta and the lensing properties are in Cardone 2004

        beta = 0.3
        self.n_sersic = 1.0/beta
        self.b = 1.0/beta

    ###############################################################################################

    def function(self, x, y, R_sersic, k_eff, center_x=0, center_y=0):
        """
        :param x: x-coordinate
        :param y: y-coordinate
        :param n_sersic: Sersic index
        :param R_sersic: half light radius
        :param k_eff: convergence at half light radius
        :param center_x: x-center
        :param center_y: y-center
        :return:
        """

        n = self.n_sersic; b = self.b
        x_red = self._x_reduced(x, y, n, R_sersic, center_x, center_y)
        hyper2f2_bx = util.hyper2F2_array(2.0*n, 2.0*n, 1.0+2.0*n, 1.0+2.0*n, -b*x_red)
        #hyper2f2_b = util.hyper2F2_array(2.0*n, 2.0*n, 1.0+2.0*n, 1.0+2.0*n, -b)
        f_eff = np.exp(b) * R_sersic**2 / 2.0 * k_eff  #* hyper2f2_b

        return f_eff * x_red**(2.0 * n) * hyper2f2_bx #/ hyper2f2_b

    ###############################################################################################

    def derivatives(self, x, y, R_sersic, k_eff, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """

        n = self.n_sersic; b = self.b
        x_ = x - center_x; y_ = y - center_y

        r = np.sqrt(x_ ** 2 + y_ ** 2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s

        alpha = -self.alpha_abs(x, y, n, R_sersic, k_eff, center_x, center_y, b_n = b)

        f_x = alpha * x_ / r
        f_y = alpha * y_ / r

        return f_x, f_y

    ###############################################################################################

    def hessian(self, x, y, n_sersic, R_sersic, k_eff, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """

        x_ = x - center_x; y_ = y - center_y

        r = np.sqrt(x_ ** 2 + y_ ** 2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s

        d_alpha_dr = self.d_alpha_dr(x, y, n_sersic, R_sersic, k_eff, center_x, center_y, b_n = self.b)
        alpha = -self.alpha_abs(x, y, n_sersic, R_sersic, k_eff, center_x, center_y, b_n = self.b)

        f_xx = -(d_alpha_dr / r + alpha / r ** 2) * x_ ** 2 / r + alpha / r
        f_yy = -(d_alpha_dr / r + alpha / r ** 2) * y_ ** 2 / r + alpha / r
        f_xy = -(d_alpha_dr / r + alpha / r ** 2) * x_ * y_ / r

        return f_xx, f_xy, f_xy, f_yy
