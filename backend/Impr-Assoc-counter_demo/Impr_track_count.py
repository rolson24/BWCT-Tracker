# Requirements: matplotlib scipy Pillow numpy prettytable easydict scikit-learn pyyaml yacs termcolor tabulate tensorboard opencv-python pyyaml yacs termcolor scikit-learn tabulate gdown faiss-gpu
# more reqs: fastreid lap cython_bbox supervision torch ultralytics

import fastreid

import plotly.graph_objects as go

import tensorflow as tf
import pandas as pd

import torch

import numpy as np
import matplotlib.pyplot as plt

import cv2

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()


import supervision as sv
print("supervision.__version__:", sv.__version__)


import torch.nn.functional as F

import cupy as cp
# from torch.backends import cudnn

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
# from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch

# cudnn.benchmark = True

def cp_array_from_cv_cuda_gpumat(mat: cv2.cuda.GpuMat) -> cp.ndarray:
    class CudaArrayInterface:
        def __init__(self, gpu_mat: cv2.cuda.GpuMat):
            w, h = gpu_mat.size()
            type_map = {
                cv2.CV_8U: "|u1",
                cv2.CV_8S: "|i1",
                cv2.CV_16U: "<u2", cv2.CV_16S: "<i2",
                cv2.CV_32S: "<i4",
                cv2.CV_32F: "<f4", cv2.CV_64F: "<f8",
            }
            self.__cuda_array_interface__ = {
                "version": 3,
                "shape": (h, w, gpu_mat.channels()) if gpu_mat.channels() > 1 else (h, w),
                "typestr": type_map[gpu_mat.depth()],
                "descr": [("", type_map[gpu_mat.depth()])],
                "stream": 1,
                "strides": (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1()) if gpu_mat.channels() > 1
                else (gpu_mat.step, gpu_mat.elemSize()),
                "data": (gpu_mat.cudaPtr(), False),
            }
    arr = cp.asarray(CudaArrayInterface(mat))

    return arr

def gpu_mat_to_torch_tensor_reid(gpu_mat):
    # Convert GpuMat to CuPy array
    # cupy_array = cp.asarray(GpuMatWrapper(gpu_mat))

    # print("before gpumat to cupy")
    # in bgr
    # gpu_mat_32f = cv2.cuda.GpuMat(gpu_mat.size(), cv2.CV_32FC3)
    # gpu_mat.convertTo(cv2.CV_32FC3, gpu_mat_32f)

    # print(f"gpu_mat_32f: {gpu_mat_32f.download()}")

    cupy_array = cp_array_from_cv_cuda_gpumat(gpu_mat)
    # print("after gpumat to cupy")

    # assert cupy_array.__cuda_array_interface__['data'][0] == b.__cuda_array_interface__['data'][0]

    # cupy_array.resize([640, 640])
    # Convert BGR to RGB (OpenCV uses BGR)
    cupy_array = cp.ascontiguousarray(cupy_array[:, :, ::-1])  # Assumes HWC format

    current_height = cupy_array.shape[0]
    target_height = cp.ceil(current_height / 32) * 32
    padding_height = int(target_height - current_height)
    padded_cupy_array = cp.pad(cupy_array, ((0, padding_height), (0, 0), (0, 0)), mode='constant', constant_values=0)

    # Convert CuPy array to PyTorch tensor
    # torch_tensor = torch.from_numpy(cupy_array).float().to('cuda')
    # print("before cupy to torch")
    # print(padded_cupy_array)
    # print(f"shape: {padded_cupy_array.shape}")
    # print(f"type: {padded_cupy_array.dtype}")
    # print(f"strides: {padded_cupy_array.strides}")
    test = padded_cupy_array.toDlpack()
    # print(test)

    # torch_tensor = torch.as_tensor(cupy_array, device='cuda')
    # assert torch_tensor.__cuda_array_interface__['data'][0] == cupy_array.__cuda_array_interface__['data'][0]



    torch_tensor = torch.from_dlpack(padded_cupy_array)
    # print("made it through cupy to torch")

    # Normalize pixel values to [0, 1]
    # torch_tensor = torch_tensor.div(255.0)

    # Permute dimensions to BCHW format
    torch_tensor = torch_tensor.permute(2, 0, 1)
    # print(torch_tensor.shape)

    return torch_tensor

def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()

    return cfg


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


class FastReIDInterface:
    def __init__(self, config_file, weights_path, device, batch_size=8):
        super(FastReIDInterface, self).__init__()
        if device != 'cpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(self.device)

        self.batch_size = batch_size

        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])

        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(weights_path)

        if self.device != 'cpu':
            self.model = self.model.eval().to(device='cuda').half()
            print(f"model is on gpu")
        else:
            self.model = self.model.eval()

        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def inference(self, image, detections):

        if detections is None or np.size(detections) == 0:
            return []

        if isinstance(image, cv2.cuda_GpuMat):
          H = image.size()[-1]
          W = image.size()[-2]
        else:
          H, W, _ = np.shape(image)

        batch_patches = []
        patches = []
        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            if isinstance(image, cv2.cuda_GpuMat):
              # print(f"image shape: {image.size()}")
              patch = cv2.cuda_GpuMat((tlbr[3] - tlbr[1], tlbr[2]-tlbr[0]), cv2.CV_8UC3)
              patch_resize = cv2.cuda_GpuMat(tuple(self.cfg.INPUT.SIZE_TEST[::-1]), cv2.CV_8UC3)
              # patch = image.roi((tlbr[0], tlbr[1], tlbr[2]-tlbr[0], tlbr[3]-tlbr[1]))
              patch = image.rowRange(tlbr[1], tlbr[3]).colRange(tlbr[0], tlbr[2])
              cv2.cuda.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), dst = patch_resize, interpolation=cv2.INTER_LINEAR)
              patch_resize_32f = cv2.cuda_GpuMat(tuple(self.cfg.INPUT.SIZE_TEST[::-1]), cv2.CV_32FC3)
              patch_resize.convertTo(cv2.CV_32FC3, patch_resize_32f)

              patch = gpu_mat_to_torch_tensor_reid(patch_resize)
              # patch = torch.as_tensor(patch_resize_32f).transpose(2, 0, 1)

              # print(f"patch on gpu? {patch.is_cuda}")
              # print(f"image shape: {patch.shape}")
              # the model expects RGB inputs
              # patch = patch[:, :, ::-1]
              patch = patch.half()
              # print(f"patch on gpu? {patch.is_cuda}")
              # print(f"gpu patch: {patch} shape: {patch.shape}")
            else:
              patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]

              # the model expects RGB inputs
              patch = patch[:, :, ::-1]

              # Apply pre-processing to image.
              patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_LINEAR)
              # patch, scale = preprocess(patch, self.cfg.INPUT.SIZE_TEST[::-1])

              # plt.figure()
              # plt.imshow(patch)
              # plt.show()

              # Make shape with a new batch dimension which is adapted for network input
              patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
              patch = patch.to(device=self.device).half()
              # print(f"cpu patch: {patch} shape: {patch.shape}")
            patches.append(patch)

            if (d + 1) % self.batch_size == 0:
                # print("new batch")
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        features = np.zeros((0, 2048))
        # features = np.zeros((0, 768))

        for patches in batch_patches:

            # Run model
            patches_ = torch.clone(patches)
            # print(f"patches: {patches} shape: {patches.shape}")
            # print(f"patches on gpu? {patches.is_cuda}")

            pred = self.model(patches)
            pred[torch.isinf(pred)] = 1.0

            feat = postprocess(pred)

            nans = np.isnan(np.sum(feat, axis=1))
            if np.isnan(feat).any():
                for n in range(np.size(nans)):
                    if nans[n]:
                        # patch_np = patches[n, ...].squeeze().transpose(1, 2, 0).cpu().numpy()
                        patch_np = patches_[n, ...]
                        patch_np_ = torch.unsqueeze(patch_np, 0)
                        pred_ = self.model(patch_np_)

                        patch_np = torch.squeeze(patch_np).cpu()
                        patch_np = torch.permute(patch_np, (1, 2, 0)).int()
                        patch_np = patch_np.numpy()

                        plt.figure()
                        plt.imshow(patch_np)
                        plt.show()

            features = np.vstack((features, feat))

        return features


import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, w, h, vx, vy, vw, vh

    contains the bounding box center position (x, y), width w, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, w, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self, conf_thresh, cov_alpha):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        self.conf_thresh = conf_thresh
        self.cov_alpha = cov_alpha


    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, w, h) with center position (x, y),
            width w, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, conf):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        # implement NK from ConfTrack paper
        # print(conf)
        innovation_cov = innovation_cov * (1 - conf) * self.cov_alpha

        mean = np.dot(self._update_mat, mean)

        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement, conf):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, w, h), where (x, y)
            is the center position, w the width, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance, conf)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T

        # # implement CW from ConfTrack paper
        # if conf < self.conf_thresh:
        #   conf_cost = 1-conf
        #   measurement = measurement + (projected_mean - measurement) * conf_cost

        innovation = measurement - projected_mean # ~yk in GIAO paper

        new_mean = mean + np.dot(innovation, kalman_gain.T)

        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')
            

import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
# from tracker import kalman_filter


def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious

def d_iou_distance(atracks, btracks):
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    # print(_ious.shape)

    atlbrs = np.tile(np.atleast_2d(np.ascontiguousarray(atlbrs, dtype=float)), len(btracks)).reshape((len(atracks), len(btracks), 4))
    # print(f"len atracks: {len(atracks)}")
    # print(f"shape atlbrs: {atlbrs.shape}")
    btlbrs = np.tile(np.atleast_2d(np.ascontiguousarray(btlbrs, dtype=float)), len(atracks)).reshape((len(btracks), len(atracks), 4))
    # print(f"len btracks: {len(btracks)}")
    # print(f"shape btlbrs: {btlbrs.shape}")
    if _ious.size == 0:
        return _ious

    a_centers = (atlbrs[:,:,2:] - atlbrs[:,:,:2])/2 + atlbrs[:,:,:2]
    b_centers = np.swapaxes((btlbrs[:,:,2:] - btlbrs[:,:,:2])/2 + btlbrs[:,:,:2], 0, 1)
    # print(f"a_centers: {a_centers}")
    # print(f"b_centers: {b_centers}")

	# calc the euclidean dist between a's and b's centers
    diff_vect = a_centers - b_centers
    # print(f"diff: {diff_vect}")
    center_dist_sq = np.sum(np.square(diff_vect), axis=2)
    # print(f"sq: {center_dist_sq}")


    top_left = np.minimum(atlbrs[:,:, :2], np.swapaxes(btlbrs[:,:, :2], 0, 1))
    # print(f"top left: {top_left}")
    bottom_right = np.maximum(atlbrs[:,:, 2:], np.swapaxes(btlbrs[:,:, 2:], 0, 1))
    # print(f"bot right: {bottom_right}")
    # calc the diagonal length from the very top left to the very bottom right
    diff_vect = bottom_right - top_left
    # print(f"diff of corners: {diff_vect}")
    outside_dist_sq = np.sum(np.square(diff_vect), axis=2)
    # print(f"sq corners: {outside_dist_sq}")
	# extra loss term to add to iou. D_IOU
    r = center_dist_sq/outside_dist_sq
    # print(f"r values: {r}")


    return 1-_ious+r



def tlbr_expand(tlbr, scale=1.2):
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    # print(f"shape of atlbrs: {np.ascontiguousarray(atlbrs, dtype=float)}")
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    # print(f"shape of track_features: {track_features.shape}")
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
    
'''BaseTrack'''  
import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0
        


'''Impr-Assoc Track'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# from tracker import matching
# from tracker.gmc import GMC
from supervision.tracker.byte_tracker.basetrack import TrackState
from supervision.detection.core import Detections

# from tracker.kalman_filter import KalmanFilter

# from fastreid import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter(0.6, 10)

    def __init__(self, tlwh, score, class_id, feat=None, feat_history=15):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        # hold on to start location for counting
        self.start_tlwh = self._tlwh

        self.score = score
        self.tracklet_len = 0

        self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(class_id, score)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov



    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # if frame_id == 1:
        # from OAI track, no unconfirmed tracks.
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh), self.score)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)


    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.score = new_track.score
        # print(f"{self.track_id}'s score: {self.score}")

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh), self.score)

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.update_cls(new_track.cls, new_track.score)


    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

def detections2boxes(detections: Detections) -> np.ndarray:
    """
    Convert Supervision Detections to numpy tensors for further computation.
    Args:
        detections (Detections): Detections/Targets in the format of sv.Detections.
        features (ndarray): The corresponding image features of each detection.
        Has shape [N, num_features]
    Returns:
        (np.ndarray): Detections as numpy tensors as in
            `(x_min, y_min, x_max, y_max, confidence, class_id, feature_vect)` order.
    """
    return np.hstack(
        (
            detections.xyxy,
            detections.confidence[:, np.newaxis],
            detections.class_id[:, np.newaxis]
        )
    )


class ImprAssocTrack:
    def __init__(self,
                 track_high_thresh=0.6,
                 track_low_thresh=0.1,
                 new_track_thresh=0.7,
                 tent_conf_thresh=0.7,
                 match_thresh=0.65, # bigger?
                 second_match_thresh=0.19,
                 overlap_thresh=0.55,
                 iou_weight=0.2,
                 track_buffer=35,
                 proximity_thresh=0.1,
                 appearance_thresh=0.25,
                 with_reid=True,
                 fast_reid_config=r"/content/drive/MyDrive/YOLO_detections/ReID/sbs_S50.yml", #need to download
                 fast_reid_weights=r"/content/drive/MyDrive/YOLO_detections/ReID/mot17_sbs_S50.pth", #need to download
                 device="gpu",

                 frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        # self.args = args

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh

        # self.tent_conf_thresh = tent_conf_thresh

        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh

        self.overlap_thresh = overlap_thresh

        self.iou_weight = iou_weight

        # self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = track_buffer
        self.kalman_filter = KalmanFilter(0.6, 10)

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        self.with_reid = with_reid

        if with_reid:
            self.encoder = FastReIDInterface(fast_reid_config, fast_reid_weights, device)

        # self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

    def update_with_detections(self, detections: Detections, img) -> Detections:
        """
        Updates the tracker with the provided detections and
            returns the updated detection results.

        Parameters:
            detections: The new detections to update with.
            img: The image for extracting features for Re-ID
        Returns:
            Detection: The updated detection results that now include tracking IDs.
        Example:
            ```python
            >>> import supervision as sv
            >>> from ultralytics import YOLO

            >>> model = YOLO(...)
            >>> byte_tracker = sv.ByteTrack()
            >>> annotator = sv.BoxAnnotator()

            >>> def callback(frame: np.ndarray, index: int) -> np.ndarray:
            ...     results = model(frame)[0]
            ...     detections = sv.Detections.from_ultralytics(results)
            ...     detections = byte_tracker.update_with_detections(detections)
            ...     labels = [
            ...         f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            ...         for _, _, confidence, class_id, tracker_id
            ...         in detections
            ...     ]
            ...     return annotator.annotate(scene=frame.copy(),
            ...                               detections=detections, labels=labels)

            >>> sv.process_video(
            ...     source_path='...',
            ...     target_path='...',
            ...     callback=callback
            ... )
            ```
        """
        tensors = detections2boxes(detections)
        # print(f"tensors: {tensors}")
        tracks = self.update_with_tensors(
            # maybe extract features here
            tensors,
            img
        )
        detections = Detections.empty()
        if len(tracks) > 0:
            detections.xyxy = np.array(
                [track.tlbr for track in tracks], dtype=np.float32
            )
            detections.class_id = np.array(
                [int(t.cls) for t in tracks], dtype=int
            )
            detections.tracker_id = np.array(
                [int(t.track_id) for t in tracks], dtype=int
            )
            detections.confidence = np.array(
                [t.score for t in tracks], dtype=np.float32
            )
        else:
            detections.tracker_id = np.array([], dtype=int)

        return detections

    def update_with_tensors(self, tensors, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(tensors):
            if tensors.shape[1] == 6: # need to validate
                scores = tensors[:, 4]
                bboxes = tensors[:, :4]
                classes = tensors[:, -1]
                # print(f"scores: {scores}")
            else:
                scores = tensors[:, 4] * tensors[:, 5]
                bboxes = tensors[:, :4]  # x1y1x2y2
                classes = tensors[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

            # print(f"high dets: {dets}")
        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        # print(f"first bboxes: {bboxes}")
        if self.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cl, f) for
                              (tlbr, s, cl, f) in zip(dets, scores_keep, classes_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cl) for
                              (tlbr, s, cl) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        low_tent = []
        high_tent = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
                # # implement LM from ConfTrack paper
                # if track.score < self.tent_conf_thresh:
                #   low_tent.append(track)
                # else:
                #   high_tent.append(track)
            else:
                tracked_stracks.append(track)

        '''Improved Association: First they calc the cost matrix of the high
        detections(func_1 -> cost_h), then the calc the cost matrix of the low
        detections (func_2 -> cost_l) and get the max values of both. Then
        B = det_h_max / det_l_max.
        Finally they calc cost = concat(cost_h, B*cost_l) for the matching
        '''

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # print(f"strack pool: {strack_pool}")
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        # print(f"high_tent: {high_tent}")

        # from ConfTrack
        # strack_pool = joint_stracks(strack_pool, high_tent) # LM algorithm
        # strack_u = joint_stracks(strack_pool, low_tent)

        # no camera motion adjustment because our device is stationary
        # # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        d_ious_dists = d_iou_distance(strack_pool, detections)
        ious = 1 - iou_distance(strack_pool, detections)
        # print(f"first dets: {detections}")
        # print(f"ious_dists: {ious_dists}")
        ious_mask = (ious < self.proximity_thresh) # o_min in ImprAssoc paper

        # if not self.args.mot20:
        # ious_dists = fuse_score(ious_dists, detections)

        '''ignore this for now. Actually i think i need to integrate this otherwise
        improved association paper isn't going to be very good. Main innovation
        is combined matching, but with no Re-ID the distance functions are the
        same.'''

        if self.with_reid:
            # ConfTrack version
            # emb_dists = embedding_distance(strack_pool, detections) / 2.0
            # raw_emb_dists = emb_dists.copy()
            # emb_dists[emb_dists > self.appearance_thresh] = 1.0
            # emb_dists[ious_dists_mask] = 1.0
            # dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0

            # Improved Association Version (CD)
            emb_dists = embedding_distance(strack_pool, detections) # high dets
            dists = self.iou_weight*d_ious_dists + (1-self.iou_weight)*emb_dists
            dists[ious_mask] = self.match_thresh + 0.00001
        else:
            dists = d_ious_dists
            dists[ious_mask] = self.match_thresh + 0.00001
        # print(f"dist first: {dists}, {len(dists)}")

        # Associate with low score detection boxes
        if len(scores):
            inds_high = scores < self.track_high_thresh
            inds_low = scores > self.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cl) for
                                 (tlbr, s, cl) in zip(dets_second, scores_second, classes_second)]
        else:
            detections_second = []
        # print(f"dets_second: {detections_second}")
        dists_second = iou_distance(strack_pool, detections_second)
        dists_second_mask = (dists_second > self.second_match_thresh) # this is what the paper used
        dists_second[dists_second_mask] = self.second_match_thresh + 0.00001
        # print(f"dists_second: {dists_second}")
        # if len(dists) == 0 or len(dists[0]) == 0:
        #     d_h_max = 1
        # else:
        #     # print(f"dists first test: {dists}")
        #     d_h_max = np.max(dists)
        # if len(dists_second) == 0 or len(dists_second[0]) == 0:
        #     d_l_max = 1
        # else:
        #     d_l_max = np.max(dists_second)
        # # print(f"max low: {d_l_max}")
        # # print(f"max high: {d_h_max}")
        # B = d_h_max/d_l_max
        B = self.match_thresh/self.second_match_thresh
        # print(f"B: {B}")

        combined_dists = np.concatenate((dists, B*dists_second), axis=1) # need to double check
        # print(f"combined_dists: {combined_dists}")

        matches, track_conf_remain, det_remain = linear_assignment(combined_dists, thresh=self.match_thresh)

        # concat detections so that it all works
        detections = np.concatenate((detections, detections_second), axis=0) # double check
        # print(f"combined_dets: {detections}")
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        '''Deal with lost tracks'''
        # for it in track_tent_remain:
        #     track = stracks_low_tent_valid[it]
        #     track.mark_removed()
        #     removed_stracks.append(track)
        # left over confirmed tracks get lost
        for it in track_conf_remain:
            # print(f"size of stracks_conf_remain: {len(stracks_conf_remain)}")
            # print(f"index: {it}")
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # detections = [detections[i] for i in u_detection]
        # ious_dists = iou_distance(unconfirmed, detections)
        # ious_dists_mask = (ious_dists > self.proximity_thresh)
        # if not self.args.mot20:
        #     ious_dists = fuse_score(ious_dists, detections)

        # if self.args.with_reid:
        #     emb_dists = embedding_distance(unconfirmed, detections) / 2.0
        #     raw_emb_dists = emb_dists.copy()
        #     emb_dists[emb_dists > self.appearance_thresh] = 1.0
        #     emb_dists[ious_dists_mask] = 1.0
        #     dists = np.minimum(ious_dists, emb_dists)
        # else:
        #     dists = ious_dists

        # matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        # for itracked, idet in matches:
        #     unconfirmed[itracked].update(detections[idet], self.frame_id)
        #     activated_starcks.append(unconfirmed[itracked])

        '''now do OAI from improved assoc'''
        # calc the iou between every unmatched det and all tracks if the max iou
        # for a det D is above overlap_thresh, discard it.
        # print(f"det remain: {det_remain}")
        sdet_remain = [detections[i] for i in det_remain]
        bboxes = [track.tlbr for track in sdet_remain]
        bboxes = np.array(bboxes)
        # print(f"bboxes: {bboxes}")
        if self.with_reid:
            features = self.encoder.inference(img, bboxes)

        unmatched_overlap = 1 - iou_distance(strack_pool, sdet_remain)

        for det_ind in range(unmatched_overlap.shape[1]): # loop over the rows
            if len(unmatched_overlap[:, det_ind]) != 0:
                if np.max(unmatched_overlap[:, det_ind]) < self.overlap_thresh:
                    # now initialize it
                    track = sdet_remain[det_ind]
                    # print(f"init track {track.track_id}, conf: {track.score}")
                    if track.score > self.new_track_thresh:
                        track.activate(self.kalman_filter, self.frame_id)
                        if self.with_reid:
                            track.update_features(features[det_ind])
                        activated_starcks.append(track)
            else:
                # if no curr tracks, then init one
                track = sdet_remain[det_ind]
                # print(f"init track {track.track_id}, conf: {track.score}")
                if track.score > self.new_track_thresh:
                    track.activate(self.kalman_filter, self.frame_id)
                    if self.with_reid:
                        track.update_features(features[det_ind])
                    activated_starcks.append(track)

        '''don't do the rest of this stuff'''

        # ''' Step 3: Second association, with low score detection boxes'''
        # if len(scores):
        #     inds_high = scores < self.track_high_thresh
        #     inds_low = scores > self.track_low_thresh
        #     inds_second = np.logical_and(inds_low, inds_high)
        #     dets_second = bboxes[inds_second]
        #     scores_second = scores[inds_second]
        #     classes_second = classes[inds_second]
        # else:
        #     dets_second = []
        #     scores_second = []
        #     classes_second = []

        # # association the untrack to the low score detections
        # if len(dets_second) > 0:
        #     '''Detections'''
        #     detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cl) for
        #                          (tlbr, s, cl) in zip(dets_second, scores_second, classes_second)]
        # else:
        #     detections_second = []

        # r_tracked_stracks = [strack_pool[i] for i in track_conf_remain if strack_pool[i].state == TrackState.Tracked]
        # dists = iou_distance(r_tracked_stracks, detections_second)
        # matches, track_conf_remain, det_low_remain = linear_assignment(dists, thresh=0.5)
        # for itracked, idet in matches:
        #     track = r_tracked_stracks[itracked]
        #     det = detections_second[idet]
        #     if track.state == TrackState.Tracked:
        #         track.update(det, self.frame_id)
        #         activated_starcks.append(track)
        #     else:
        #         track.re_activate(det, self.frame_id, new_id=False)
        #         refind_stracks.append(track)


        # # implement LM from ConfTrack paper
        # '''Step 4: low-confidence track matching with high-conf dets'''
        # # Associate with high score detection boxes
        # stracks_conf_remain = [r_tracked_stracks[i] for i in track_conf_remain]
        # ious_dists = iou_distance(low_tent, stracks_conf_remain)
        # _, low_tent_valid, _ = linear_assignment(ious_dists, thresh=1-0.7) # want to get rid of tracks with low iou costs
        # stracks_low_tent_valid = [low_tent[i] for i in low_tent_valid]
        # stracks_det_high_remain = [detections[i] for i in det_high_remain]
        # C_low_ious = iou_distance(stracks_low_tent_valid, stracks_det_high_remain)
        # ious_dists_mask = (C_low_ious > self.proximity_thresh)

        # if self.with_reid:
        #     emb_dists = embedding_distance(stracks_low_tent_valid, stracks_det_high_remain) / 2.0
        #     raw_emb_dists = emb_dists.copy()
        #     emb_dists[emb_dists > self.appearance_thresh] = 1.0
        #     emb_dists[ious_dists_mask] = 1.0
        #     dists = np.minimum(C_low_ious, emb_dists)
        # else:
        #     dists = C_low_ious

        # matches, track_tent_remain, det_high_remain = linear_assignment(dists, thresh=0.3) # need to find this val in ConfTrack paper

        # for itracked, idet in matches:
        #     low_tent[itracked].update(stracks_det_high_remain[idet], self.frame_id)
        #     activated_starcks.append(low_tent[itracked])






        # """ Step 5: Init new stracks"""
        # # u_detection = [*det_high_remain, *det_low_remain]
        # for inew in det_high_remain:
        #     track = stracks_det_high_remain[inew]
        #     # print(f"init new track {track.track_id} with score :{track.score}")
        #     if track.score < self.new_track_thresh:
        #         continue

        #     track.activate(self.kalman_filter, self.frame_id)
        #     activated_starcks.append(track)

        # for inew in det_low_remain:
        #     track = detections_second[inew]
        #     if track.score < self.new_track_thresh:
        #         continue

        #     track.activate(self.kalman_filter, self.frame_id)
        #     activated_starcks.append(track)

        """ Step 6: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
    
    
'''Color transfer'''
# import the necessary packages
import numpy as np
import cv2

# before calling run image_stats(source_image)

def color_transfer(source_img_stats, target, clip=True, preserve_paper=True):
	"""
	Transfers the color distribution from the source to the target
	image using the mean and standard deviations of the L*a*b*
	color space.

	This implementation is (loosely) based on to the "Color Transfer
	between Images" paper by Reinhard et al., 2001.

	Parameters:
	-------
	source_img_stats: list of NumPy arrays
		extracted image stats from the source image.
    (lMean, lStd, aMean, aStd, bMean, bStd)
	target: NumPy array
		OpenCV image in BGR color space (the target image)
	clip: Should components of L*a*b* image be scaled by np.clip before
		converting back to BGR color space?
		If False then components will be min-max scaled appropriately.
		Clipping will keep target image brightness truer to the input.
		Scaling will adjust image brightness to avoid washed out portions
		in the resulting color transfer that can be caused by clipping.
	preserve_paper: Should color transfer strictly follow methodology
		layed out in original paper? The method does not always produce
		aesthetically pleasing results.
		If False then L*a*b* components will scaled using the reciprocal of
		the scaling factor proposed in the paper.  This method seems to produce
		more consistently aesthetically pleasing results

	Returns:
	-------
	transfer: NumPy array
		OpenCV image (w, h, 3) NumPy array (uint8)
	"""
	# convert the images from the RGB to L*ab* color space, being
	# sure to utilizing the floating point data type (note: OpenCV
	# expects floats to be 32-bit, so use that instead of 64-bit)
	# source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

	# compute color statistics for the source and target images
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = source_img_stats
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

	# subtract the means from the target image
	(l, a, b) = cv2.split(target)
	l -= lMeanTar
	a -= aMeanTar
	b -= bMeanTar

	if preserve_paper:
		# scale by the standard deviations using paper proposed factor
		l = (lStdTar / lStdSrc) * l
		a = (aStdTar / aStdSrc) * a
		b = (bStdTar / bStdSrc) * b
	else:
		# scale by the standard deviations using reciprocal of paper proposed factor
		l = (lStdSrc / lStdTar) * l
		a = (aStdSrc / aStdTar) * a
		b = (bStdSrc / bStdTar) * b

	# add in the source mean
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc

	# clip/scale the pixel intensities to [0, 255] if they fall
	# outside this range
	l = _scale_array(l, clip=clip)
	a = _scale_array(a, clip=clip)
	b = _scale_array(b, clip=clip)

	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

	# return the color transferred image
	return transfer

def image_stats(image):
	"""
	Parameters:
	-------
	image: NumPy array
		OpenCV image in L*a*b* color space

	Returns:
	-------
	Tuple of mean and standard deviations for the L*, a*, and b*
	channels, respectively
	"""
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())

	# return the color statistics
	return (lMean, lStd, aMean, aStd, bMean, bStd)

def _min_max_scale(arr, new_range=(0, 255)):
	"""
	Perform min-max scaling to a NumPy array

	Parameters:
	-------
	arr: NumPy array to be scaled to [new_min, new_max] range
	new_range: tuple of form (min, max) specifying range of
		transformed array

	Returns:
	-------
	NumPy array that has been scaled to be in
	[new_range[0], new_range[1]] range
	"""
	# get array's current min and max
	mn = arr.min()
	mx = arr.max()

	# check if scaling needs to be done to be in new_range
	if mn < new_range[0] or mx > new_range[1]:
		# perform min-max scaling
		scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
	else:
		# return array if already in range
		scaled = arr

	return scaled

def _scale_array(arr, clip=True):
	"""
	Trim NumPy array values to be in [0, 255] range with option of
	clipping or scaling.

	Parameters:
	-------
	arr: array to be trimmed to [0, 255] range
	clip: should array be scaled by np.clip? if False then input
		array will be min-max scaled to range
		[max([arr.min(), 0]), min([arr.max(), 255])]

	Returns:
	-------
	NumPy array that has been scaled to be in [0, 255] range
	"""
	if clip:
		scaled = np.clip(arr, 0, 255)
	else:
		scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
		scaled = _min_max_scale(arr, new_range=scale_range)

	return scaled
	
	
	
''' load YOLOv8'''
from ultralytics import YOLO
# change this
MODEL = "/content/drive/MyDrive/Workspace/YOLO_detections/runs/detect/train7/weights/best.pt"


yolo_model = YOLO(MODEL)
yolo_model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = yolo_model.model.names

# class_ids of interest - pedestrians
selected_classes = [0,1,2,3]
print(CLASS_NAMES_DICT)
            
            
            
'''Settings'''
# settings
import supervision as sv

num_lines = 4

# in script extract these coords from args
line_ends = []
LINE_1_START = sv.Point(300, 360)
LINE_1_END = sv.Point(700, 360)
line_ends.append((LINE_1_START, LINE_1_END))
LINE_2_START = sv.Point(300, 360)
LINE_2_END = sv.Point(300, 720) # ?
line_ends.append((LINE_2_START, LINE_2_END))

LINE_3_START = sv.Point(300, 720)
LINE_3_END = sv.Point(1280, 550)
line_ends.append((LINE_3_START, LINE_3_END))

LINE_4_START = sv.Point(1280, 550)
LINE_4_END = sv.Point(700, 360)
line_ends.append((LINE_4_START, LINE_4_END))

# change these
SOURCE_VIDEO_PATH = "/content/drive/MyDrive/Workspace/2024-01-26_foco.mp4"
TARGET_VIDEO_PATH_annotated = "/content/drive/MyDrive/Workspace/census_tool_2024-01-26_foco.mp4(color calibrated Impr_Assoc output).mp4"
TARGET_VIDEO_PATH_clean = "/content/drive/MyDrive/Workspace/census_tool_2024-01-26_foco.mp4(color calibrated_for annotation).mp4"
COLOR_SOURCE_IMG = "/content/drive/MyDrive/Workspace/color_ref_img_1_21_300.jpg"

DATA_OUTPUT = "/content/drive/MyDrive/Workspace/census_tool_2024-01-26_foco(Impr_Assoc track output).csv"


''' run the tracker '''
from numpy.random.mtrand import randint
from torch import rand
import csv
# create ConfTracker instance
impr_assoc_tracker = ImprAssocTrack()

# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

out = cv2.VideoWriter(TARGET_VIDEO_PATH_clean,cv2.VideoWriter_fourcc(*'MP4V'), 30, (video_info.width,video_info.height))


# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, end=10)

line_zones = []
for end_points in line_ends:
  line_zones.append(sv.LineZone(start=end_points[0], end=end_points[1]))

# create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

# create instance of TraceAnnotator
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=500) # 20s*25fps = 500

# create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

fps_monitor = sv.FPSMonitor()

'''for cpu version'''
# Color source image to correct colors CPU
color_source = cv2.imread(COLOR_SOURCE_IMG)
color_source = cv2.cvtColor(color_source, cv2.COLOR_BGR2LAB).astype("float32")
source_img_stats = image_stats(color_source)

'''for gpu version'''
# Color source image to correct colors GPU
# color_source = cv2.imread(COLOR_SOURCE_IMG)
# color_source = cv2.resize(color_source, (video_info.width, video_info.height), interpolation=cv2.INTER_LINEAR)
# gpu_color_source = cv2.cuda_GpuMat()
# gpu_color_source.upload(color_source)
# cv2.cuda.cvtColor(gpu_color_source, cv2.COLOR_BGR2LAB, gpu_color_source)
# source_img_stats = image_stats_gpu(gpu_color_source)

# gpu_frame = cv2.cuda_GpuMat()
# # gpu_frame_resize = cv2.cuda_GpuMat()
# # yolo_input_res = [640, 640]

paths = []
prev_removed_tracks = []

line_counts = []
for i in range(len(line_zones)):
  class_counts = {}
  for val in CLASS_NAMES_DICT.values():
    class_counts[val+"_in"] = 0
    class_counts[val+"_out"] = 0
  line_counts.append(class_counts)

# define call back function to be used in video processing
def callback(frame: np.ndarray, index:int) -> np.ndarray:
    # model prediction on single frame and conversion to supervision Detections
    print(index)
    # results = model.predict(frame, confidence=40, overlap=30).json()
    # detections = sv.Detections.from_roboflow(results)

    ''' Color calibration '''
    # frame[:,:,2] = frame[:,:,2]*.5


    # # control Contrast by 1.5
    # alpha = .7
    # # control brightness by 50
    # beta = 30
    # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # frame[:,:,1] = frame[:,:,1]*1.2
    # frame[:,:,0] = frame[:,:,0]*1.05

    # # Create the sharpening kernel
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # # Sharpen the image
    # frame = cv2.filter2D(frame, -1, kernel)

    '''Color Calibration with color_transfer'''
    frame = color_transfer(source_img_stats, frame, clip=False, preserve_paper=False)

    out.write(frame)
    '''Color Calibration with color_transfer_gpu'''
    # gpu_frame.upload(frame)
    # # cv2.cuda.resize(gpu_frame, yolo_input_res, gpu_frame_resize)
    # '''Color Calibration with color_transfer'''
    # frame_gpuMat = color_transfer_gpu(source_img_stats, gpu_frame, clip=False, preserve_paper=False)
    # # print("before transfer torch to tensor")
    # frame_tensor = gpu_mat_to_torch_tensor(frame_gpuMat)
    # print("made it to here")

    results = yolo_model(frame, verbose=False, iou=0.7, conf=0.1)[0]
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above
    detections = detections[np.isin(detections.class_id, selected_classes)]
    print(detections)

    # tracking detections
    detections = impr_assoc_tracker.update_with_detections(detections, frame)

    with open(DATA_OUTPUT, 'a+', newline='', encoding='UTF8') as f:
      writer = csv.writer(f)
      for track, _, conf, class_id, tracker_id, _ in detections:
        writer.writerow([index, tracker_id, track[0], track[1], track[2]-track[0], track[3]-track[1], conf, -1, -1, -1])

    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id, _
        in detections
    ]
    # print(detections)
    if detections.tracker_id is None:
      print(detections)
      annotated_frame = frame.copy()
      # annotated_frame = frame_gpuMat.download()

      # detections.tracker_id = [randint(0, 5000)]
    else:
      annotated_frame = trace_annotator.annotate(
          scene=frame.copy(),
          # scene = frame_gpuMat.download(),

          detections=detections
      )
      # print(trace_annotator.trace.get(detections.tracker_id[0]))
      # this gets the pixel vals of all of the points each object has been.
      for removed_track in conf_tracker.removed_stracks:
        if removed_track not in prev_removed_tracks:
          paths.append(trace_annotator.trace.get(removed_track.track_id))
          prev_removed_tracks.append(removed_track)

      annotated_frame=box_annotator.annotate(
          scene=annotated_frame,
          detections=detections,
          labels=labels)
      
    detections = detections[np.isin(detections.class_id, selected_classes)]

    # update line counter
    for i, line_zone in enumerate(line_zones):
      objects_in, objects_out = line_zone.trigger(detections)
      for obj in detections.class_id[np.isin(objects_in, True)]:
        # print(obj)
        line_counts[i][CLASS_NAMES_DICT[obj]+"_in"] += 1
      for obj in detections.class_id[np.isin(objects_out, True)]:
        line_counts[i][CLASS_NAMES_DICT[obj]+"_out"] += 1


    # return frame with box and line annotated result
    fps_monitor.tick()
    print(f"fps: {fps_monitor()}")
    for line_zone in line_zones:
      annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
    return  annotated_frame

