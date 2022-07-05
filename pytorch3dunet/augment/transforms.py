import importlib
import random

import numpy as np
import torch
import h5py
import os
from scipy.ndimage import rotate, zoom, map_coordinates, gaussian_filter, convolve
from skimage import measure
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from scipy.ndimage import percentile_filter, binary_dilation, maximum_filter, minimum_filter, generic_filter, binary_erosion, shift

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(47)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, m):
        for t in self.transforms:
            m = t(m)
        return m

class RandomScale:
    def __init__(self, random_state, sigma=[0.7, 1.3], order=0, execution_probalility=0.5, **kwargs):
        self.random_state = random_state
        self.sigma = sigma
        self.order = order
        self.execution_probability = execution_probalility

    def __call__(self, m):
        if self.random_state.uniform() > self.execution_probability:
            return m

        alpha = self.random_state.uniform(self.sigma[0], self.sigma[1])
        if alpha > 0.95 and alpha <= 1:
            alpha -= 0.05
        elif alpha > 1 and alpha <= 1.05:
            alpha += 0.05

        zoomed = zoom(m, alpha, order=self.order, mode='reflect')
        zoom_shape = np.array(zoomed.shape)
        m_shape = np.array(m.shape)

        if alpha < 1:
            ret = np.zeros(m.shape)
            start = self.random_state.randint(0, m_shape - zoom_shape, size=3)
            stop = start + zoom_shape

            ret[Slice3D(start, stop).tuple] = zoomed

            return ret
        else:
            start = self.random_state.randint(0, zoomed.shape - m_shape, size=3)
            stop = start + m_shape
            return zoomed[Slice3D(start, stop).tuple]

class RandomTranslate:
    def __init__(self, random_state, sigma=[0, 64], order=0, execution_probalility=0.5, **kwargs):
        self.random_state = random_state
        self.sigma = sigma
        self.order = order
        self.execution_probability = execution_probalility

    def __call__(self, m):
        if self.random_state.uniform() > self.execution_probability:
            return m

        x = self.random_state.randint(self.sigma[0], self.sigma[1])
        y = self.random_state.randint(self.sigma[0], self.sigma[1])
        z = self.random_state.randint(self.sigma[0], self.sigma[1])

        m = shift(m, [x, y, z], order=self.order, mode='reflect')
        return m


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, axis_prob=0.5, execution_probalility=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob
        self.execution_probability = execution_probalility

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        if self.random_state.uniform() > self.execution_probability:
            return m

        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m


class Slice3D:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __sub__(self, other):
        assert isinstance(other, np.ndarray), "Must be ndarray"
        assert len(other) == 3

        return Slice3D(self.start - other, self.stop - other)

    @property
    def tuple(self):
        slices = []
        for i in range(3):
            slices.append(slice(self.start[i], self.stop[i]))

        return tuple(slices)

class ShrinkMask:

    def __init__(self, iterations, min_size, **kwargs):
        self.iterations = iterations
        self.min_size = min_size
        pass

    def __call__(self, m):
        for i in range(self.iterations):
            if m.sum() < self.min_size:
                break
            m = binary_erosion(m, iterations=1)
        return m

class ArteryMask:

    def __init__(self, pmax, **kwargs):
        self.pmax = pmax

    def __call__(self, m):
        m = m[None]
        img_b = m.copy()
        img_b[img_b >= self.pmax] = 1
        img_b[img_b < self.pmax] = 0
        img_b = minimum_filter(img_b, size=2)
        img_b = binary_dilation(img_b, iterations=3)
        
        ret = np.concatenate([m, img_b])
        return ret


class AneuInsertion:

    def __init__(
            self,
            random_state,
            path,
            padding=10,
            iters=1,
            type="raw",
            crop=True,
            execution_probalility=0.5,
            **kwargs
    ):
        assert type in ["raw", "label"]

        self.execution_probability = execution_probalility
        self.random_state = random_state
        self.path = path
        self.iters = iters
        self.type = type
        self.padding = padding
        self.cases = os.listdir(self.path)
        self.crop = crop

    def load_rand_case(self):
        pick = self.random_state.choice(self.cases)
        return h5py.File(f"{self.path}/{pick}", mode="r")

    @staticmethod
    def pad_min_max(max, ival, padding):
        return slice(
            np.clip(ival[0] - padding, a_min=0, a_max=max - 1),
            np.clip(ival[1] + padding, a_min=0, a_max=max - 1)
        )

    def get_aneurysm_bounds(self, mask):
        x_s = mask.sum(axis=(1, 2))
        y_s = mask.sum(axis=(0, 2))
        z_s = mask.sum(axis=(0, 1))

        x = np.where(x_s)[0][[0, -1]]
        y = np.where(y_s)[0][[0, -1]]
        z = np.where(z_s)[0][[0, -1]]

        x = self.pad_min_max(mask.shape[0], x, self.padding)
        y = self.pad_min_max(mask.shape[1], y, self.padding)
        z = self.pad_min_max(mask.shape[2], z, self.padding)

        return x, y, z

    def __call__(self, m):
        if self.random_state.uniform() > self.execution_probability:
            return m

        for _ in range(self.iters):
            insert_case = self.load_rand_case()
            bound = self.get_aneurysm_bounds(insert_case["label"][:])
            insert_case_raw = insert_case[self.type][:]

            if self.type == "raw":
                insert_case_raw /= insert_case_raw.max()

            aneu = insert_case_raw[bound]

            # Rotating and scaling of aneurysm
            rotation = self.random_state.randint(0, 360)
            axes = [(1, 0), (2, 1), (2, 0)]
            ax = axes[self.random_state.randint(0, 3)]
            aneu = rotate(aneu, rotation, axes=ax, reshape=False, order=0)

            # Get Zoom between 0.8 and 1.2
            zoom_fac = self.random_state.random() * 0.4 + 0.8
            aneu = zoom(aneu, zoom_fac, order=0)

            aneu_shape = np.array(aneu.shape)
            m_shape = np.array(m.shape)
            # Random aneurysm to big -> skip insertion
            if (aneu_shape >= m_shape).any():
                return m

            if self.crop:
                # Allow aneurysm to be placed slightly outside the area of the current case
                # but at least 50 % of the aneurysm should be inside.
                aneu_shape_h = aneu_shape // 2
                pos = self.random_state.randint(-aneu_shape_h, m_shape - aneu_shape_h)

                start = np.max([pos, np.zeros(3)], axis=0).astype(int)
                end = np.min([pos + aneu_shape, m_shape], axis=0).astype(int)

                m_slice = Slice3D(start, end)
                aneu_slice = m_slice - pos

                m[m_slice.tuple] = np.maximum(aneu[aneu_slice.tuple], m[m_slice.tuple])
            else:
                # get Position
                start = self.random_state.randint(0, m_shape - aneu_shape)
                end = start + aneu_shape
                m_slice = Slice3D(start, end)

                m[m_slice.tuple] = np.maximum(aneu, m[m_slice.tuple])

        return m


class Perlin:

    def __init__(
            self,
            random_state,
            alpha,
            res,
            shape,
            execution_probalility=0.5,
            **kwargs
    ):
        self.random_state = random_state
        self.alpha = alpha
        self.res = res
        self.shape = shape
        self.execution_probability = execution_probalility


    def generate_perlin_noise_3d(self):
        def f(t):
            return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

        delta = (self.res[0] / self.shape[0], self.res[1] / self.shape[1], self.res[2] / self.shape[2])
        d = (self.shape[0] // self.res[0], self.shape[1] // self.res[1], self.shape[2] // self.res[2])
        grid = np.mgrid[0:self.res[0]:delta[0], 0:self.res[1]:delta[1], 0:self.res[2]:delta[2]]
        grid = grid.transpose(1, 2, 3, 0) % 1
        # Gradients
        theta = 2 * np.pi * self.random_state.rand(self.res[0] + 1, self.res[1] + 1, self.res[2] + 1)
        phi = 2 * np.pi * self.random_state.rand(self.res[0] + 1, self.res[1] + 1, self.res[2] + 1)
        gradients = np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=3)
        gradients[-1] = gradients[0]
        g000 = gradients[0:-1, 0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g100 = gradients[1:, 0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g010 = gradients[0:-1, 1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g110 = gradients[1:, 1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g001 = gradients[0:-1, 0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g101 = gradients[1:, 0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g011 = gradients[0:-1, 1:, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g111 = gradients[1:, 1:, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        # Ramps
        n000 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g000, 3)
        n100 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g100, 3)
        n010 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g010, 3)
        n110 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g110, 3)
        n001 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g001, 3)
        n101 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g101, 3)
        n011 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g011, 3)
        n111 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g111, 3)
        # Interpolation
        t = f(grid)
        n00 = n000 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n100
        n10 = n010 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n110
        n01 = n001 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n101
        n11 = n011 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n111
        n0 = (1 - t[:, :, :, 1]) * n00 + t[:, :, :, 1] * n10
        n1 = (1 - t[:, :, :, 1]) * n01 + t[:, :, :, 1] * n11
        return ((1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1)

    def __call__(self, x):
        if self.random_state.uniform() < self.execution_probability:
            noise = self.generate_perlin_noise_3d()
            n, l, k = x.shape
            return x + self.alpha * noise[:n, :l, :k]

        return x


class Insertion:

    def __init__(
            self,
            random_state,
            path,
            min_patch_size=(32, 32, 32),
            max_patch_size=(128, 128, 128),
            iters=1,
            type="raw",
            execution_probalility=0.5,
            **kwargs
    ):
        assert type in ["raw", "label"]

        self.random_state = random_state
        self.path = path
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.iters = iters
        self.type = type
        self.cases = os.listdir(self.path)
        self.execution_probability = execution_probalility

    def load_rand_case(self):
        pick = self.random_state.choice(self.cases)
        return h5py.File(f"{self.path}/{pick}", mode="r")

    def __call__(self, x):
        if self.execution_probability > self.random_state.uniform():
            return x

        for _ in range(self.iters):
            insert_case_raw = self.load_rand_case()[self.type][:]

            if self.type == "raw":
                insert_case_raw /= insert_case_raw.max()

            # Get random size
            size = self.random_state.randint(low=self.min_patch_size, high=np.array(self.max_patch_size), size=3)

            # Get position to insert in image a
            insert_pos = self.random_state.randint(low=0, high=np.array(x.shape) - size, size=3)

            # Get position to use from image b
            retrieve_pos = self.random_state.randint(low=0, high=np.array(insert_case_raw.shape) - size, size=3)

            insert_slice = Slice3D(insert_pos, insert_pos + size)
            retrieve_slice = Slice3D(retrieve_pos, retrieve_pos + size)

            x[insert_slice.tuple] = insert_case_raw[retrieve_slice.tuple]

        return x


# Just for debugging
class Identity:
    def __call__(self, x):
        return x


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, **kwargs):
        self.random_state = random_state
        # always rotate around z-axis
        self.axis = (1, 2)

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, self.axis)
        else:
            channels = [np.rot90(m[c], k, self.axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=30, axes=None, mode='reflect', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m

class Threshold:
    def __init__(self, low, high, **kwargs):
        self.low = low
        self.high = high

    def __call__(self, m):
        m[m < self.low] = 0
        m[m > self.high] = 1

        return m

class PercentileThreshold:
    def __init__(self, low, high, **kwargs):
        self.low = low
        self.high = high

    def __call__(self, m):
        low_perc = np.percentile(m, self.low)
        high_perc = np.percentile(m, self.high)

        m[m < low_perc] = 0
        m[m > high_perc] = 1

        return m

class RandomContrast:
    """
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(self, random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (m - self.mean)
            return np.clip(result, -1, 1)

        return m


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=0 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=2000, sigma=50, execution_probability=0.1, apply_3d=True,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim in [3, 4]

            if m.ndim == 3:
                volume_shape = m.shape
            else:
                volume_shape = m[0].shape

            if self.apply_3d:
                dz = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="reflect") * self.alpha
            else:
                dz = np.zeros_like(m)

            dy, dx = [
                gaussian_filter(
                    self.random_state.randn(*volume_shape),
                    self.sigma, mode="reflect"
                ) * self.alpha for _ in range(2)
            ]

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx

            if m.ndim == 3:
                return map_coordinates(m, indices, order=self.spline_order, mode='reflect')
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
                return np.stack(channels, axis=0)

        return m


class CropToFixed:
    def __init__(self, random_state, size=(256, 256), centered=False, **kwargs):
        self.random_state = random_state
        self.crop_y, self.crop_x = size
        self.centered = centered

    def __call__(self, m):
        def _padding(pad_total):
            half_total = pad_total // 2
            return (half_total, pad_total - half_total)

        def _rand_range_and_pad(crop_size, max_size):
            """
            Returns a tuple:
                max_value (int) for the corner dimension. The corner dimension is chosen as `self.random_state(max_value)`
                pad (int): padding in both directions; if crop_size is lt max_size the pad is 0
            """
            if crop_size < max_size:
                return max_size - crop_size, (0, 0)
            else:
                return 1, _padding(crop_size - max_size)

        def _start_and_pad(crop_size, max_size):
            if crop_size < max_size:
                return (max_size - crop_size) // 2, (0, 0)
            else:
                return 0, _padding(crop_size - max_size)

        assert m.ndim in (3, 4)
        if m.ndim == 3:
            _, y, x = m.shape
        else:
            _, _, y, x = m.shape

        if not self.centered:
            y_range, y_pad = _rand_range_and_pad(self.crop_y, y)
            x_range, x_pad = _rand_range_and_pad(self.crop_x, x)

            y_start = self.random_state.randint(y_range)
            x_start = self.random_state.randint(x_range)

        else:
            y_start, y_pad = _start_and_pad(self.crop_y, y)
            x_start, x_pad = _start_and_pad(self.crop_x, x)

        if m.ndim == 3:
            result = m[:, y_start:y_start + self.crop_y, x_start:x_start + self.crop_x]
            return np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode='reflect')
        else:
            channels = []
            for c in range(m.shape[0]):
                result = m[c][:, y_start:y_start + self.crop_y, x_start:x_start + self.crop_x]
                channels.append(np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode='reflect'))
            return np.stack(channels, axis=0)


class AbstractLabelToBoundary:
    AXES_TRANSPOSE = [
        (0, 1, 2),  # X
        (0, 2, 1),  # Y
        (2, 0, 1)  # Z
    ]

    def __init__(self, ignore_index=None, aggregate_affinities=False, append_label=False, **kwargs):
        """
        :param ignore_index: label to be ignored in the output, i.e. after computing the boundary the label ignore_index
            will be restored where is was in the patch originally
        :param aggregate_affinities: aggregate affinities with the same offset across Z,Y,X axes
        :param append_label: if True append the orignal ground truth labels to the last channel
        :param blur: Gaussian blur the boundaries
        :param sigma: standard deviation for Gaussian kernel
        """
        self.ignore_index = ignore_index
        self.aggregate_affinities = aggregate_affinities
        self.append_label = append_label

    def __call__(self, m):
        """
        Extract boundaries from a given 3D label tensor.
        :param m: input 3D tensor
        :return: binary mask, with 1-label corresponding to the boundary and 0-label corresponding to the background
        """
        assert m.ndim == 3

        kernels = self.get_kernels()
        boundary_arr = [np.where(np.abs(convolve(m, kernel)) > 0, 1, 0) for kernel in kernels]
        channels = np.stack(boundary_arr)
        results = []
        if self.aggregate_affinities:
            assert len(kernels) % 3 == 0, "Number of kernels must be divided by 3 (one kernel per offset per Z,Y,X axes"
            # aggregate affinities with the same offset
            for i in range(0, len(kernels), 3):
                # merge across X,Y,Z axes (logical OR)
                xyz_aggregated_affinities = np.logical_or.reduce(channels[i:i + 3, ...]).astype(np.int32)
                # recover ignore index
                xyz_aggregated_affinities = _recover_ignore_index(xyz_aggregated_affinities, m, self.ignore_index)
                results.append(xyz_aggregated_affinities)
        else:
            results = [_recover_ignore_index(channels[i], m, self.ignore_index) for i in range(channels.shape[0])]

        if self.append_label:
            # append original input data
            results.append(m)

        # stack across channel dim
        return np.stack(results, axis=0)

    @staticmethod
    def create_kernel(axis, offset):
        # create conv kernel
        k_size = offset + 1
        k = np.zeros((1, 1, k_size), dtype=np.int32)
        k[0, 0, 0] = 1
        k[0, 0, offset] = -1
        return np.transpose(k, axis)

    def get_kernels(self):
        raise NotImplementedError


class StandardLabelToBoundary:
    def __init__(self, ignore_index=None, append_label=False, mode='thick', foreground=False,
                 **kwargs):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.mode = mode
        self.foreground = foreground

    def __call__(self, m):
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2, mode=self.mode)
        boundaries = boundaries.astype('int32')

        results = []
        if self.foreground:
            foreground = (m > 0).astype('uint8')
            results.append(_recover_ignore_index(foreground, m, self.ignore_index))

        results.append(_recover_ignore_index(boundaries, m, self.ignore_index))

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


class BlobsWithBoundary:
    def __init__(self, mode=None, append_label=False, **kwargs):
        if mode is None:
            mode = ['thick', 'inner', 'outer']
        self.mode = mode
        self.append_label = append_label

    def __call__(self, m):
        assert m.ndim == 3

        # get the segmentation mask
        results = [(m > 0).astype('uint8')]

        for bm in self.mode:
            boundary = find_boundaries(m, connectivity=2, mode=bm)
            results.append(boundary)

        if self.append_label:
            results.append(m)

        return np.stack(results, axis=0)


class BlobsToMask:
    """
    Returns binary mask from labeled image, i.e. every label greater than 0 is treated as foreground.

    """

    def __init__(self, append_label=False, boundary=False, cross_entropy=False, **kwargs):
        self.cross_entropy = cross_entropy
        self.boundary = boundary
        self.append_label = append_label

    def __call__(self, m):
        assert m.ndim == 3

        # get the segmentation mask
        mask = (m > 0).astype('uint8')
        results = [mask]

        if self.boundary:
            outer = find_boundaries(m, connectivity=2, mode='outer')
            if self.cross_entropy:
                # boundary is class 2
                mask[outer > 0] = 2
                results = [mask]
            else:
                results.append(outer)

        if self.append_label:
            results.append(m)

        return np.stack(results, axis=0)


class RandomLabelToAffinities(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the max_offset (thickness) of the border. Then the offset is picked at random every time you call
    the transformer (offset is picked form the range 1:max_offset) for each axis and the boundary computed.
    One may use this scheme in order to make the network more robust against various thickness of borders in the ground
    truth  (think of it as a boundary denoising scheme).
    """

    def __init__(self, random_state, max_offset=10, ignore_index=None, append_label=False, z_offset_scale=2, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label, aggregate_affinities=False)
        self.random_state = random_state
        self.offsets = tuple(range(1, max_offset + 1))
        self.z_offset_scale = z_offset_scale

    def get_kernels(self):
        rand_offset = self.random_state.choice(self.offsets)
        axis_ind = self.random_state.randint(3)
        # scale down z-affinities due to anisotropy
        if axis_ind == 2:
            rand_offset = max(1, rand_offset // self.z_offset_scale)

        rand_axis = self.AXES_TRANSPOSE[axis_ind]
        # return a single kernel
        return [self.create_kernel(rand_axis, rand_offset)]


class LabelToAffinities(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels (which can be seen
    as an affinity graph: https://arxiv.org/pdf/1706.00120.pdf)
    One specify the offsets (thickness) of the border. The boundary will be computed via the convolution operator.
    """

    def __init__(self, offsets, ignore_index=None, append_label=False, aggregate_affinities=False, z_offsets=None,
                 **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label,
                         aggregate_affinities=aggregate_affinities)

        assert isinstance(offsets, list) or isinstance(offsets, tuple), 'offsets must be a list or a tuple'
        assert all(a > 0 for a in offsets), "'offsets must be positive"
        assert len(set(offsets)) == len(offsets), "'offsets' must be unique"
        if z_offsets is not None:
            assert len(offsets) == len(z_offsets), 'z_offsets length must be the same as the length of offsets'
        else:
            # if z_offsets is None just use the offsets for z-affinities
            z_offsets = list(offsets)
        self.z_offsets = z_offsets

        self.kernels = []
        # create kernel for every axis-offset pair
        for xy_offset, z_offset in zip(offsets, z_offsets):
            for axis_ind, axis in enumerate(self.AXES_TRANSPOSE):
                final_offset = xy_offset
                if axis_ind == 2:
                    final_offset = z_offset
                # create kernels for a given offset in every direction
                self.kernels.append(self.create_kernel(axis, final_offset))

    def get_kernels(self):
        return self.kernels


class LabelToZAffinities(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels (which can be seen
    as an affinity graph: https://arxiv.org/pdf/1706.00120.pdf)
    One specify the offsets (thickness) of the border. The boundary will be computed via the convolution operator.
    """

    def __init__(self, offsets, ignore_index=None, append_label=False, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label)

        assert isinstance(offsets, list) or isinstance(offsets, tuple), 'offsets must be a list or a tuple'
        assert all(a > 0 for a in offsets), "'offsets must be positive"
        assert len(set(offsets)) == len(offsets), "'offsets' must be unique"

        self.kernels = []
        z_axis = self.AXES_TRANSPOSE[2]
        # create kernels
        for z_offset in offsets:
            self.kernels.append(self.create_kernel(z_axis, z_offset))

    def get_kernels(self):
        return self.kernels


class LabelToBoundaryAndAffinities:
    """
    Combines the StandardLabelToBoundary and LabelToAffinities in the hope
    that that training the network to predict both would improve the main task: boundary prediction.
    """

    def __init__(self, xy_offsets, z_offsets, append_label=False, blur=False, sigma=1, ignore_index=None, mode='thick',
                 foreground=False, **kwargs):
        # blur only StandardLabelToBoundary results; we don't want to blur the affinities
        self.l2b = StandardLabelToBoundary(blur=blur, sigma=sigma, ignore_index=ignore_index, mode=mode,
                                           foreground=foreground)
        self.l2a = LabelToAffinities(offsets=xy_offsets, z_offsets=z_offsets, append_label=append_label,
                                     ignore_index=ignore_index)

    def __call__(self, m):
        boundary = self.l2b(m)
        affinities = self.l2a(m)
        return np.concatenate((boundary, affinities), axis=0)


class LabelToMaskAndAffinities:
    def __init__(self, xy_offsets, z_offsets, append_label=False, background=0, ignore_index=None, **kwargs):
        self.background = background
        self.l2a = LabelToAffinities(offsets=xy_offsets, z_offsets=z_offsets, append_label=append_label,
                                     ignore_index=ignore_index)

    def __call__(self, m):
        mask = m > self.background
        mask = np.expand_dims(mask.astype(np.uint8), axis=0)
        affinities = self.l2a(m)
        return np.concatenate((mask, affinities), axis=0)


class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    """

    def __init__(self, eps=1e-10, mean=None, std=None, channelwise=False, **kwargs):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise

    def __call__(self, m):
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per-channel
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)


class PercentileNormalizer:
    def __init__(self, pmin=1, pmax=99, channelwise=False, eps=1e-10, **kwargs):
        self.eps = eps
        self.pmin = pmin
        self.pmax = pmax
        self.channelwise = channelwise

    def __call__(self, m):
        if self.channelwise:
            axes = list(range(m.ndim))
            # average across channels
            axes = tuple(axes[1:])
            pmin = np.percentile(m, self.pmin, axis=axes, keepdims=True)
            pmax = np.percentile(m, self.pmax, axis=axes, keepdims=True)
        else:
            pmin = np.percentile(m, self.pmin)
            pmax = np.percentile(m, self.pmax)

        return (m - pmin) / (pmax - pmin + self.eps)


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, m):
        min_val = m.min()
        max_val = m.max()

        norm_0_1 = (m - min_val) / (max_val - min_val)
        return norm_0_1

class HardThreshold:
    def __init__(self, **kwargs):
        pass
        
    def __call__(self, m):
        m[m<0.4]=0
        kernel = np.asarray([[[0.125,0.25,125],
                    [0.25,0.5,0.25],
                    [0.125,0.25,0.125]],[[0.25,0.5,0.25],
                    [0.5,1,0.5],
                    [0.25,0.5,0.25]],
                    [[0.125,0.25,125],
                    [0.25,0.5,0.25],
                    [0.125,0.25,0.125]]])
                    
        np.convolve(m, kernel, mode='full')
        return m


class AdditiveGaussianNoise:
    def __init__(self, random_state, scale=(0.0, 1.0), sigma=[0, 0.3], execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.scale = scale
        self.sigma = sigma

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            std = self.random_state.uniform(self.scale[0], self.scale[1])
            gaussian_noise = self.random_state.normal(0, std, size=m.shape)
            alpha = self.random_state.uniform(self.sigma[0], self.sigma[1])
            return m + alpha * gaussian_noise
        return m


class AdditivePoissonNoise:
    def __init__(self, random_state, lam=(0.0, 1.0), sigma=[0.1, 0.5], execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.lam = lam
        self.sigma = sigma

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            lam = self.random_state.uniform(self.lam[0], self.lam[1])
            poisson_noise = self.random_state.poisson(lam, size=m.shape)

            alpha = self.random_state.uniform(self.sigma[0], self.sigma[1])
            return m + alpha * poisson_noise
        return m


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class Relabel:
    """
    Relabel a numpy array of labels into a consecutive numbers, e.g.
    [10, 10, 0, 6, 6] -> [2, 2, 0, 1, 1]. Useful when one has an instance segmentation volume
    at hand and would like to create a one-hot-encoding for it. Without a consecutive labeling the task would be harder.
    """

    def __init__(self, append_original=False, run_cc=True, ignore_label=None, **kwargs):
        self.append_original = append_original
        self.ignore_label = ignore_label
        self.run_cc = run_cc

        if ignore_label is not None:
            assert append_original, "ignore_label present, so append_original must be true, so that one can localize the ignore region"

    def __call__(self, m):
        orig = m
        if self.run_cc:
            # assign 0 to the ignore region
            m = measure.label(m, background=self.ignore_label)

        _, unique_labels = np.unique(m, return_inverse=True)
        result = unique_labels.reshape(m.shape)
        if self.append_original:
            result = np.stack([result, orig])
        return result


class Identity:
    def __init__(self, **kwargs):
        pass

    def __call__(self, m):
        return m


class RgbToLabel:
    def __call__(self, img):
        img = np.array(img)
        assert img.ndim == 3 and img.shape[2] == 3
        result = img[..., 0] * 65536 + img[..., 1] * 256 + img[..., 2]
        return result


class LabelToTensor:
    def __call__(self, m):
        m = np.array(m)
        return torch.from_numpy(m.astype(dtype='int64'))


class GaussianBlur3D:
    def __init__(self, sigma=[.1, 2.], execution_probability=0.5, **kwargs):
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, x):
        if random.random() < self.execution_probability:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = gaussian(x, sigma=sigma)
            return x
        return x


class Transformer:
    def __init__(self, phase_config, base_config):
        self.phase_config = phase_config
        self.config_base = base_config
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)

    def raw_transform(self):
        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    def weight_transform(self):
        return self._create_transform('weight')

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('pytorch3dunet.augment.transforms')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        config['random_state'] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input
