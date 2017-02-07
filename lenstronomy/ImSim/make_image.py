__author__ = 'sibirrer'

import astrofunc.util as util
from astrofunc.util import Util_class
from astrofunc.LensingProfiles.shapelets import Shapelets
from astrofunc.LensingProfiles.gaussian import Gaussian

from lenstronomy.ImSim.lens_model import LensModel
from lenstronomy.ImSim.source_model import SourceModel
from lenstronomy.ImSim.lens_light_model import LensLightModel
from lenstronomy.DeLens.de_lens import DeLens

import scipy.ndimage as ndimage
import scipy.signal as signal
import numpy as np
import copy


class MakeImage(object):
    """
    this class uses functions of lens_model and source_model to make a lensed image
    """
    def __init__(self, kwargs_options, kwargs_data=None, kwargs_psf=None):
        self.LensModel = LensModel(kwargs_options)
        self.SourceModel = SourceModel(kwargs_options)
        self.LensLightModel = LensLightModel(kwargs_options, kwargs_data)
        self.DeLens = DeLens()
        self.kwargs_options = kwargs_options
        self.kwargs_data = kwargs_data
        self.kwargs_psf = kwargs_psf
        self.util_class = Util_class()
        self.gaussian = Gaussian()

        if kwargs_data is not None and 'sigma_background' in kwargs_data and 'image_data' in kwargs_data and 'exp_time' in kwargs_data:
            sigma_b = kwargs_data['sigma_background']
            exp_map = kwargs_data.get('exposure_map', None)
            if exp_map is not None:
                exp_map[exp_map <= 0] = 10**(-3)
                f = util.image2array(exp_map)
            else:
                f = kwargs_data['exp_time']
            data = kwargs_data['image_data']
            self.C_D = self.DeLens.get_covariance_matrix(util.image2array(data), sigma_b, f)
        self.shapelets = Shapelets()
        if kwargs_options['source_type'] == 'SERSIC':
            from astrofunc.LightProfiles.sersic import Sersic
            self.sersic = Sersic()
        elif kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
            from astrofunc.LightProfiles.sersic import Sersic_elliptic
            self.sersic = Sersic_elliptic()
        try:
            self.ra_coords = kwargs_data.get('x_coords', None)
            self.dec_coords = kwargs_data.get('y_coords', None)
        except:
            pass

    def mapping_IS(self, x, y, kwargs_else=None, **kwargs):
        """
        maps image to source position (inverse deflection)
        """
        dx, dy = self.LensModel.alpha(x, y, kwargs_else, **kwargs)
        return x - dx, y - dy

    def map_coord2pix(self, ra, dec):
        """

        :param ra: ra coordinates, relative
        :param dec: dec coordinates, relative
        :param x_0: pixel value in x-axis of ra,dec = 0,0
        :param y_0: pixel value in y-axis of ra,dec = 0,0
        :param M:
        :return:
        """
        x_0 = self.kwargs_data['zero_point_x']
        y_0 = self.kwargs_data['zero_point_y']
        M = self.kwargs_data['transform_angle2pix']
        return util.map_coord2pix(ra, dec, x_0, y_0, M)

    def get_surface_brightness(self, x, y, **kwargs):
        """
        returns the surface brightness of the source at coordinate x, y
        """
        I_xy = self.SourceModel.surface_brightness(x, y, **kwargs)
        return I_xy

    def get_lens_all(self, x, y, kwargs_else=None, **kwargs):
        """
        returns all the lens properties
        :return:
        """
        potential, alpha1, alpha2, kappa, gamma1, gamma2, mag = self.LensModel.all(x, y, kwargs_else, **kwargs)
        return potential, alpha1, alpha2, kappa, gamma1, gamma2, mag

    def psf_convolution(self, grid, grid_scale, **kwargs):
        """
        convolves a given pixel grid with a PSF
        """
        if self.kwargs_options['psf_type'] == 'gaussian':
            sigma = kwargs['sigma']/grid_scale
            if 'truncate' in kwargs:
                sigma_truncate = kwargs['truncate']
            else:
                sigma_truncate = 3.
            img_conv = ndimage.filters.gaussian_filter(grid, sigma, mode='nearest', truncate=sigma_truncate)
            return img_conv
        elif self.kwargs_options['psf_type'] == 'pixel':
            kernel = kwargs['kernel']
            if 'kernel_fft' in kwargs:
                kernel_fft = kwargs['kernel_fft']
                img_conv1 = self.util_class.fftconvolve(grid, kernel, kernel_fft, mode='same')
            else:
                img_conv1 = signal.fftconvolve(grid, kernel, mode='same')
            return img_conv1
        return grid

    def re_size_convolve(self, image, numPix, deltaPix, subgrid_res, kwargs_psf, unconvolved=False):
        gridScale = deltaPix/subgrid_res
        if self.kwargs_options['psf_type'] == 'pixel':
            grid_re_sized = self.util_class.re_size(image, numPix)
            if unconvolved:
                grid_final = grid_re_sized
            else:
                grid_final = self.psf_convolution(grid_re_sized, gridScale, **kwargs_psf)
        elif self.kwargs_options['psf_type'] == 'NONE':
            grid_final = self.util_class.re_size(image, numPix)
        else:
            if unconvolved:
                grid_conv = image
            else:
                grid_conv = self.psf_convolution(image, gridScale, **kwargs_psf)
            grid_final = self.util_class.re_size(grid_conv, numPix)
        return grid_final

    def add_noise2image(self, image):
        """
        adds Poisson and Gaussian noise to the modeled image
        :param image:
        :return:
        """
        gaussian = util.add_background(image, self.kwargs_data["sigma_background"])
        poisson = util.add_poisson(image, self.kwargs_data.get("exposure_map", self.kwargs_data["exp_time"]))
        image_noisy = image + gaussian + poisson
        return image_noisy

    def reduced_residuals(self, model, error_map=0):
        """

        :param model:
        :return:
        """
        residual = (model - self.kwargs_data["image_data"])/np.sqrt(util.array2image(self.C_D)+np.abs(error_map))*self.kwargs_data["mask"]
        return residual

    def _update_linear_kwargs(self, param, kwargs_source, kwargs_lens_light):
        """
        links linear parameters to kwargs arguments
        :param param:
        :return:
        """
        if not self.kwargs_options['source_type'] == 'NONE':
            kwargs_source['I0_sersic'] = param[0]
            i = 1
        else:
            i = 0
        kwargs_lens_light['I0_sersic'] = param[i]
        if self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            kwargs_lens_light['I0_3'] = param[i+1]
            kwargs_lens_light['I0_2'] = param[i+2]
        return kwargs_source, kwargs_lens_light

    def make_image_ideal(self, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, inv_bool=False, no_lens=False):
        map_error = self.kwargs_options.get('error_map', False)
        num_order = self.kwargs_options.get('shapelet_order', 0)
        if no_lens is True:
            x_source, y_source = x_grid, y_grid
        else:
            x_source, y_source = self.mapping_IS(x_grid, y_grid, kwargs_else, **kwargs_lens)
        mask = self.kwargs_data['mask']
        A, error_map, _ = self.get_response_matrix(x_grid, y_grid, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, num_order, mask, map_error=map_error, shapelets_off=self.kwargs_options.get('shapelets_off', False))
        data = self.kwargs_data['image_data']
        d = util.image2array(data*mask)
        param, cov_param, wls_model = self.DeLens.get_param_WLS(A.T, 1/(self.C_D+error_map), d, inv_bool=inv_bool)
        grid_final = util.array2image(wls_model)
        kwargs_source, kwargs_lens_light = self._update_linear_kwargs(param, kwargs_source, kwargs_lens_light)
        if map_error is True:
             error_map = util.array2image(error_map)
        else:
            error_map = np.zeros_like(grid_final)
        return grid_final, error_map, cov_param, param

    def make_image_ideal_noMask(self, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, inv_bool=False, unconvolved=False):
        map_error = self.kwargs_options.get('error_map', False)
        num_order = self.kwargs_options.get('shapelet_order', 0)
        x_source, y_source = self.mapping_IS(x_grid, y_grid, kwargs_else, **kwargs_lens)
        mask = self.kwargs_data['mask']
        A, error_map, _ = self.get_response_matrix(x_grid, y_grid, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, num_order, mask, map_error=map_error, shapelets_off=self.kwargs_options.get('shapelets_off', False))
        A_pure, _, _ = self.get_response_matrix(x_grid, y_grid, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, num_order, mask=1, map_error=map_error, shapelets_off=self.kwargs_options.get('shapelets_off', False), unconvolved=unconvolved)
        data = self.kwargs_data['image_data']
        d = util.image2array(data*mask)
        param, cov_param, wls_model = self.DeLens.get_param_WLS(A.T, 1/(self.C_D+error_map), d, inv_bool=inv_bool)
        image_pure = A_pure.T.dot(param)
        grid_final = util.array2image(image_pure)
        return grid_final, param, util.array2image(error_map)

    def make_image_with_params(self, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, param, num_order):
        """
        make a image with a realisation of linear parameter values "param"
        """
        map_error = self.kwargs_options.get('error_map', False)
        x_source, y_source = self.mapping_IS(x_grid, y_grid, kwargs_else, **kwargs_lens)
        A, error_map, bool_string = self.get_response_matrix(x_grid, y_grid, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, num_order, mask=1, map_error=map_error, shapelets_off=self.kwargs_options.get('shapelets_off', False), unconvolved=True)
        image_pure = A.T.dot(param*bool_string)
        image_ = A.T.dot(param*(1-bool_string))
        image_conv = self.psf_convolution(util.array2image(image_pure), deltaPix/subgrid_res, **self.kwargs_psf)
        image_ = util.array2image(image_)
        return image_conv + image_, util.array2image(error_map)

    def make_image_surface_extended_source(self, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_else, numPix, deltaPix, subgrid_res):
        x_source, y_source = self.mapping_IS(x_grid, y_grid, kwargs_else, **kwargs_lens)
        I_xy = self.get_surface_brightness(x_source, y_source, **kwargs_source)
        grid = util.array2image(I_xy)
        grid_final = self.re_size_convolve(grid, numPix, deltaPix, subgrid_res, self.kwargs_psf)
        return grid_final

    def make_image_lens_light(self, x_grid, y_grid, kwargs_lens_light, numPix, deltaPix, subgrid_res):
        mask = self.kwargs_data['mask_lens_light']
        lens_light_response = self.get_lens_light_response(x_grid, y_grid, kwargs_lens_light)
        n_lens_light = len(lens_light_response)
        n = 0
        A = np.zeros((n_lens_light, numPix ** 2))
        for i in range(0, n_lens_light):
            image = util.array2image(lens_light_response[i])
            image = self.re_size_convolve(image, numPix, deltaPix, subgrid_res, self.kwargs_psf)
            A[n, :] = util.image2array(image*mask)
            n += 1
        data = self.kwargs_data['image_data']
        d = util.image2array(data*mask)
        param, cov_param, wls_model = self.DeLens.get_param_WLS(A.T, 1/self.C_D, d, inv_bool=False)
        grid_final = util.array2image(wls_model)
        return grid_final, cov_param, param

    def get_lens_surface_brightness(self, x_grid, y_grid, numPix, deltaPix, subgrid_res, kwargs_lens_light):
        lens_light = self.LensLightModel.surface_brightness(x_grid, y_grid, **kwargs_lens_light)
        lens_light = util.array2image(lens_light)
        lens_light_final = self.re_size_convolve(lens_light, numPix, deltaPix, subgrid_res, self.kwargs_psf)
        return lens_light_final

    def _matrix_configuration(self, x_grid, y_grid, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, num_order, shapelets_off=False):
        if self.kwargs_options['source_type'] == 'NONE':
            n_source = 0
        else:
            n_source = 1
        if self.kwargs_options.get('point_source', False):
            if self.kwargs_options.get('psf_iteration', False):
                n_points = len(kwargs_psf['kernel_list'])
            else:
                n_points = len(kwargs_else['ra_pos'])
        else:
            n_points = 0
        if self.kwargs_options['lens_light_type'] == 'NONE':
            n_lens_light = 0
            lens_light_response = []
        else:
            lens_light_response = self.get_lens_light_response(x_grid, y_grid, kwargs_lens_light)
            n_lens_light = len(lens_light_response)
        if shapelets_off:
            n_shapelets = 0
        else:
            n_shapelets = (num_order+2)*(num_order+1)/2
        if self.kwargs_options.get("clump_enhance", False):
            num_order_enhance = self.kwargs_options.get('num_order_clump', 1)
            num_enhance = (num_order_enhance+2)*(num_order_enhance+1)/2
        else:
            num_enhance = 0
        if self.kwargs_options.get("source_substructure", False):
            num_clump = kwargs_source["num_clumps"]
            num_order = kwargs_source["subclump_order"]
            numShapelets = (num_order + 2) * (num_order + 1) / 2
            num_subclump = numShapelets * num_clump
        else:
            num_subclump = 0
        num_param = n_shapelets + n_points + n_lens_light + n_source + num_enhance + num_subclump
        if not self.kwargs_options['source_type'] == 'NONE':
            num_param += 1
        return num_param, n_source, n_lens_light, n_points, n_shapelets, lens_light_response, num_enhance, num_subclump

    def get_response_matrix(self, x_grid, y_grid, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, num_order, mask, map_error=False, shapelets_off=False, unconvolved=False):
        kwargs_psf = self.kwargs_psf
        num_param, n_source, n_lens_light, n_points, n_shapelets, lens_light_response, num_enhance, num_subclump = self._matrix_configuration(x_grid, y_grid, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, num_order, shapelets_off)
        A = np.zeros((num_param, numPix**2))
        if map_error is True:
            error_map = np.zeros((numPix, numPix))
        else:
            error_map = 0
        n = 0
        bool_string = np.ones(num_param)
        # response of sersic source profile
        if not self.kwargs_options['source_type'] == 'NONE':
            new = {'I0_sersic': 1}
            kwargs_source_new = dict(kwargs_source.items() + new.items())
            sersic_light = self.sersic.function(x_source, y_source, **kwargs_source_new)
            image = self.re_size_convolve(util.array2image(sersic_light), numPix, deltaPix, subgrid_res, kwargs_psf, unconvolved)
            A[n, :] = util.image2array(image*mask)
            n += 1
        # response of lens light profile
        for i in range(0, n_lens_light):
            image = util.array2image(lens_light_response[i])
            image = self.re_size_convolve(image, numPix, deltaPix, subgrid_res, kwargs_psf, unconvolved)
            A[n, :] = util.image2array(image*mask)
            n += 1
        # response of point sources
        if self.kwargs_options.get('point_source', False):
            A_point, error_map = self.get_psf_response(n_points, kwargs_psf, kwargs_else, mask, map_error=map_error)
            A[n:n+n_points, :] = A_point
            bool_string[n:n+n_points] = 0
            n += n_points
        # response of source shapelet coefficients
        if not shapelets_off:
            center_x = kwargs_source['center_x']
            center_y = kwargs_source['center_y']
            beta = kwargs_else['shapelet_beta']
            A_shapelets = self.get_shapelet_response(x_source, y_source, num_order, center_x, center_y, beta, kwargs_psf, numPix, deltaPix, subgrid_res, mask, unconvolved)
            A[n:n+n_shapelets, :] = A_shapelets
            n += n_shapelets
        if self.kwargs_options.get("clump_enhance", False):
            num_order_clump = self.kwargs_options.get('num_order_clump', 0)
            clump_scale = self.kwargs_options.get('clump_scale', 1)
            kwargs_else_enh = copy.deepcopy(kwargs_else)
            kwargs_else_enh["phi_E_clump"] = 0
            center_x, center_y, beta = self.position_size_estimate(kwargs_else['x_clump'], kwargs_else['y_clump'],
                                                              kwargs_lens, kwargs_else_enh, kwargs_else["r_trunc"], clump_scale)
            A_shapelets_enhance = self.get_shapelet_response(x_source, y_source, num_order_clump, center_x, center_y, beta,
                                                     kwargs_psf, numPix, deltaPix, subgrid_res, mask, unconvolved)
            A[n:n + num_enhance, :] = A_shapelets_enhance
            n += num_enhance
        if self.kwargs_options.get("source_substructure", False):
            A_subclump = self.subclump_shapelet_response(x_source, y_source, kwargs_source, kwargs_psf, deltaPix, numPix, subgrid_res, mask, unconvolved)
            A[n:n + num_subclump, :] = A_subclump
        if map_error is True:
            error_map = util.image2array(error_map)
        return A, error_map, bool_string

    def get_psf_response(self, num_param, kwargs_psf, kwargs_else, mask, map_error=False):
        """

        :param n_points:
        :param x_pos:
        :param y_pos:
        :param psf_large:
        :return: response matrix of point sources
        """
        ra_pos = kwargs_else['ra_pos']
        dec_pos = kwargs_else['dec_pos']
        x_pos, y_pos = self.map_coord2pix(ra_pos, dec_pos)
        n_points = len(x_pos)
        data = self.kwargs_data['image_data']
        psf_large = kwargs_psf['kernel_large']
        amplitudes = kwargs_else.get('point_amp', np.ones_like(x_pos))
        numPix = len(data)
        if map_error is True:
            error_map = np.zeros((numPix, numPix))
            for i in range(0, n_points):
                error_map = self.get_error_map(data, x_pos[i], y_pos[i], psf_large, amplitudes[i], error_map, kwargs_psf['error_map'])
        else:
            error_map = 0
        A = np.zeros((num_param, numPix**2))
        if self.kwargs_options.get('psf_iteration', False):
            psf_list = kwargs_psf['kernel_list']
            for k in range(num_param):
                psf = psf_list[k]
                grid2d = np.zeros((numPix, numPix))
                for i in range(0, len(x_pos)):
                    grid2d = util.add_layer2image(grid2d, x_pos[i], y_pos[i], 1, amplitudes[i]*psf)
                A[k, :] = util.image2array(grid2d*mask)
        else:
            for i in range(num_param):
                grid2d = np.zeros((numPix, numPix))
                point_source = util.add_layer2image(grid2d, x_pos[i], y_pos[i], 1, psf_large)
                A[i, :] = util.image2array(point_source*mask)
        return A, error_map

    def get_shapelet_response(self, x_source, y_source, num_order, center_x, center_y, beta, kwargs_psf, numPix, deltaPix, subgrid_res, mask, unconvolved=False):
        num_param = (num_order+1)*(num_order+2)/2
        A = np.zeros((num_param, numPix**2))
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x_source, y_source, beta, num_order, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {'center_x': center_x, 'center_y': center_y, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': 1}
            image = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            image = util.array2image(image)
            image = self.re_size_convolve(image, numPix, deltaPix, subgrid_res, kwargs_psf, unconvolved)
            response = util.image2array(image*mask)
            A[i, :] = response
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        return A

    def subclump_shapelet_response(self, x_source, y_source, kwargs_source, kwargs_psf, deltaPix, numPix, subgrid_res, mask=1, unconvolved=False):
        """
        returns response matrix for general inputs
        :param x_grid:
        :param y_grid:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_psf:
        :param kwargs_lens_light:
        :param kwargs_else:
        :param numPix:
        :param deltaPix:
        :param subgrid_res:
        :return:
        """
        num_clump = kwargs_source["num_clumps"]
        x_pos = kwargs_source["subclump_x"]
        y_pos = kwargs_source["subclump_y"]
        sigma = kwargs_source["subclump_sigma"]
        num_order = kwargs_source["subclump_order"]
        numShapelets = (num_order+2)*(num_order+1)/2
        num_param = numShapelets*num_clump
        A = np.zeros((num_param, numPix**2))
        k = 0
        for j in range(0, num_clump):
            H_x, H_y = self.shapelets.pre_calc(x_source, y_source, sigma[j], num_order, x_pos[j], y_pos[j])
            n1 = 0
            n2 = 0
            for i in range(0, numShapelets):
                kwargs_source_shapelet = {'center_x': x_pos[j], 'center_y': y_pos[j], 'n1': n1, 'n2': n2, 'beta': sigma[j], 'amp': 1}
                image = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
                image = util.array2image(image)
                image = self.re_size_convolve(image, numPix, deltaPix, subgrid_res, kwargs_psf, unconvolved)
                response = util.image2array(image*mask)
                A[k, :] = response
                if n1 == 0:
                    n1 = n2 + 1
                    n2 = 0
                else:
                    n1 -= 1
                    n2 += 1
                k += 1
        return A

    def get_lens_light_response(self, x_grid, y_grid, kwargs_lens_light):
        """
        computes the responses to all linear parameters (normalisations) in the lens light models
        :param x_grid:
        :param y_grid:
        :param kwargs_lens_light:
        :return:
        """
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
            new = {'I0_sersic': 1, 'I0_2': 1}
            kwargs_lens_light_new = dict(kwargs_lens_light.items() + new.items())
            ellipse, spherical = self.LensLightModel.func.function_split(x_grid, y_grid, **kwargs_lens_light_new)
            response = [ellipse, spherical]
        elif self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            new = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
            kwargs_lens_light_new = dict(kwargs_lens_light.items() + new.items())
            ellipse1, ellipse2, spherical = self.LensLightModel.func.function_split(x_grid, y_grid, **kwargs_lens_light_new)
            response = [ellipse1, ellipse2, spherical]
        elif self.kwargs_options['lens_light_type'] == 'SERSIC' or 'SERSIC_ELLIPSE':
            new = {'I0_sersic': 1}
            kwargs_lens_light_new = dict(kwargs_lens_light.items() + new.items())
            ellipse = self.LensLightModel.func.function(x_grid, y_grid, **kwargs_lens_light_new)
            response = [ellipse]
        elif self.kwargs_options['lens_light_type'] == 'NONE':
            response = []
        else:
            raise ValueError('lens_light_type %s not specified well' %(self.kwargs_options['lens_light_type']))
        return response

    def get_error_map(self, data, x_pos, y_pos, psf_kernel, amplitude, error_map, psf_error_map):
        if self.kwargs_options.get('fix_error_map', False):
            amp_estimated = amplitude
        else:
            amp_estimated = self.estimate_amp(data, x_pos, y_pos, psf_kernel)
        error_map = util.add_layer2image(error_map, x_pos, y_pos, 1, psf_error_map*(amp_estimated*psf_kernel)**2, key='linear')
        return error_map

    def estimate_amp(self, data, x_pos, y_pos, psf_kernel):
        """
        estimates the amplitude of a point source located at x_pos, y_pos
        :param data:
        :param x_pos:
        :param y_pos:
        :param deltaPix:
        :return:
        """
        numPix = len(data)
        #data_center = int((numPix-1.)/2)
        x_int = int(round(x_pos-0.49999))#+data_center
        y_int = int(round(y_pos-0.49999))#+data_center
        if x_int > 2 and x_int < numPix-2 and y_int > 2 and y_int < numPix-2:
            mean_image = max(np.sum(data[y_int-2:y_int+3, x_int-2:x_int+3]), 0)
            num = len(psf_kernel)
            center = int((num-0.5)/2)
            mean_kernel = np.sum(psf_kernel[center-2:center+3, center-2:center+3])
            amp_estimated = mean_image/mean_kernel
        else:
            amp_estimated = 0
        return amp_estimated

    def get_source(self, param, num_order, beta, x_grid, y_grid, kwargs_source, cov_param=None):
        """

        :param param:
        :param num_order:
        :param beta:

        :return:
        """
        if not self.kwargs_options['source_type'] == 'NONE':
            new = {'I0_sersic': param[0], 'center_x': 0, 'center_y': 0}
            kwargs_source_new = dict(kwargs_source.items() + new.items())
            source = self.get_surface_brightness(x_grid, y_grid, **kwargs_source_new)
        else:
            source = np.zeros_like(x_grid)
        num_param_shapelets = (num_order+2)*(num_order+1)/2
        shapelets = Shapelets(interpolation=False, precalc=False)
        error_map_source = np.zeros_like(x_grid)
        n1 = 0
        n2 = 0
        basis_functions = np.zeros((len(param), len(x_grid)))
        for i in range(len(param)-num_param_shapelets, len(param)):
            source += shapelets.function(x_grid, y_grid, param[i], beta, n1, n2, center_x=0, center_y=0)
            basis_functions[i, :] = shapelets.function(x_grid, y_grid, 1, beta, n1, n2, center_x=0, center_y=0)
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        if cov_param is not None:
            error_map_source = np.zeros_like(x_grid)
            for i in range(len(error_map_source)):
                error_map_source[i] = basis_functions[:, i].T.dot(cov_param).dot(basis_functions[:,i])
        return util.array2image(source), util.array2image(error_map_source)

    def get_psf(self, param, kwargs_psf, kwargs_else):
        """
        returns the psf estimates from the different basis sets
        only analysis function
        :param param:
        :param kwargs_psf:
        :return:
        """
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
            a = 2
        elif self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            a = 3
        elif self.kwargs_options['lens_light_type'] == 'SERSIC' or 'SERSIC_ELLIPSE':
            a = 1
        else:
            a = 0
        if not self.kwargs_options['source_type'] == 'NONE':
            a += 1
        kernel_list = kwargs_psf['kernel_list']
        num_param = len(kernel_list)
        A_psf, _ = self.get_psf_response(num_param, kwargs_psf, kwargs_else, mask=1, map_error=False)
        num_param = len(kernel_list)
        param_psf = param[a:a+num_param]
        psf = A_psf.T.dot(param_psf)
        return psf

    def get_cov_basis(self, A, pix_error=None):
        """
        computes covariance matrix of the response function A_i with pixel errors
        :param A: A[i,:] response of parameter i on the image (in 1d array)
        :param pix_error:
        :return:
        """
        if pix_error is None:
            pix_error = 1
        numParam = len(A)
        M = np.zeros((numParam, numParam))
        for i in range(numParam):
            M[i,:] = np.sum(A * A[i]*pix_error, axis=1)
        return M

    def get_magnification_model(self, kwargs_lens, kwargs_else):
        """
        computes the point source magnification at the position of the point source images
        :param kwargs_lens:
        :param kwargs_else:
        :return: list of magnifications
        """
        if 'ra_pos' in kwargs_else and 'dec_pos' in kwargs_else:
            ra_pos = kwargs_else['ra_pos']
            dec_pos = kwargs_else['dec_pos']
        else:
            raise ValueError('No point source positions assigned')
        mag = self.LensModel.magnification(ra_pos, dec_pos, kwargs_else, **kwargs_lens)
        return ra_pos, dec_pos, mag

    def get_image_amplitudes(self, param, kwargs_else):
        """
        returns the amplitudes of the point source images
        :param param: list of parameters determined by the least square fitting
        :return: the selected list
        """
        #i=0 source sersic
        n = len(kwargs_else['ra_pos']) # number of point sources
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
            return param[3:3+n]
        elif self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            return param[4:4+n]
        elif self.kwargs_options['lens_light_type'] == 'NONE':
            return param[1:1+n]
        else:
            print('WARNING: no suited lens light type found. Return might be corrupted.')
            return param[1:1+n]

    def get_time_delay(self, kwargs_lens, kwargs_source, kwargs_else):
        """

        :return: time delay in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """
        if 'ra_pos' in kwargs_else and 'dec_pos' in kwargs_else:
            ra_pos = kwargs_else['ra_pos']
            dec_pos = kwargs_else['dec_pos']
        else:
            raise ValueError('No point source positions assigned')
        potential = self.LensModel.potential(ra_pos, dec_pos, kwargs_else, **kwargs_lens)
        ra_source = kwargs_source['center_x']
        dec_source = kwargs_source['center_y']
        geometry = (ra_pos - ra_source)**2 + (dec_pos - dec_source)**2
        return geometry/2 - potential

    def position_size_estimate(self, ra_pos, dec_pos, kwargs_lens, kwargs_else, delta, scale=1):
        """
        estimate the magnification at the positions and define resolution limit
        :param ra_pos:
        :param dec_pos:
        :param kwargs_lens:
        :param kwargs_else:
        :return:
        """
        x, y = self.mapping_IS(ra_pos, dec_pos, kwargs_else, **kwargs_lens)
        d_x, d_y = util.points_on_circle(delta*2, 10)
        x_s, y_s = self.mapping_IS(ra_pos + d_x, dec_pos + d_y, kwargs_else, **kwargs_lens)
        x_m = np.mean(x_s)
        y_m = np.mean(y_s)
        r_m = np.sqrt((x_s - x_m) ** 2 + (y_s - y_m) ** 2)
        r_min = np.sqrt(r_m.min(axis=0)*r_m.max(axis=0))/2 * scale
        return x, y, r_min

