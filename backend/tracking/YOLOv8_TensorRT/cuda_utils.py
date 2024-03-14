from pathlib import Path
from typing import List, Tuple, Union

import torch
import cv2
import numpy as np
from numpy import ndarray
import cupy as cp


# Cuda version:
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


def gpu_mat_to_torch_tensor(gpu_mat):
    # Convert GpuMat to CuPy array
    # cupy_array = cp.asarray(GpuMatWrapper(gpu_mat))

    # print("before gpumat to cupy")
    # in bgr
    gpu_mat_32f = cv2.cuda.GpuMat(gpu_mat.size(), cv2.CV_32FC3)
    gpu_mat.convertTo(cv2.CV_32FC3, gpu_mat_32f)
    # gpu_mat.release()

    # print(f"gpu_mat_32f: {gpu_mat_32f.download()}")

    cupy_array = cp_array_from_cv_cuda_gpumat(gpu_mat_32f)
    # print("after gpumat to cupy")

    # assert cupy_array.__cuda_array_interface__['data'][0] == b.__cuda_array_interface__['data'][0]

    # cupy_array.resize([640, 640])
    # Convert BGR to RGB (OpenCV uses BGR)
    cupy_array = cp.ascontiguousarray(cupy_array[:, :, ::-1])  # Assumes HWC format

    # Pad the top and bottom to get the frame to be the correct size.
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
    torch_tensor = torch_tensor.div(255.0)

    # Permute dimensions to BCHW format
    torch_tensor = torch_tensor.permute(2, 0, 1).unsqueeze(0)
    # print(torch_tensor.shape)

    return torch_tensor

def pad_image_on_gpu_with_pytorch(image_tensor: torch.Tensor, 
                                  pad: Tuple[int, int, int, int], 
                                  padding_mode: str = 'constant', 
                                  value: int = 114) -> torch.Tensor:
    """
    Pad an image tensor on the GPU using PyTorch.

    Parameters:
    - image_tensor (torch.Tensor): The input image tensor on the GPU. Expected shape is (C, H, W).
    - pad (Tuple[int, int, int, int]): The padding to apply on each side of the image as 
      (left, right, top, bottom).
    - padding_mode (str, optional): The type of padding to apply. Can be 'constant', 'reflect', etc. 
      Default is 'constant'.
    - value (int, optional): The value to use for constant padding. Default is 114.

    Returns:
    - torch.Tensor: The padded image tensor on the GPU.

    Note:
    - Ensure the input image tensor is already transferred to the GPU.
    - This function directly utilizes PyTorch's functionality for padding and GPU computation.
    """
    return F.pad(image_tensor, pad, mode=padding_mode, value=value)

def letterbox_cuda(im: cv2.cuda_GpuMat,
                   new_shape: Union[Tuple[int, int], List[int]] = (640, 640),
                   color: Union[Tuple[int, int, int], List[int]] = (114, 114, 114)) \
        -> Tuple[cv2.cuda_GpuMat, float, Tuple[float, float]]:
    """
    Resize and pad an image to a new shape using CUDA-accelerated functions, 
    maintaining the aspect ratio of the original image. The padding is applied 
    to ensure the resized image matches the new shape while keeping its original 
    aspect ratio. This function is designed to work with images processed on a GPU.

    Parameters:
    - im (cv2.cuda_GpuMat): The input image as a GPU matrix.
    - new_shape (Union[Tuple[int, int], List[int]], optional): The target size of the image 
    as a tuple (width, height). Default is (640, 640).
    - color (Union[Tuple[int, int, int], List[int]], optional): The color of the padding 
    as a tuple (B, G, R). Default is (114, 114, 114).

    Returns:
    - Tuple[cv2.cuda_GpuMat, float, Tuple[float, float]]: A tuple containing the resized 
    and padded image as a GPU matrix, the scale ratio (new / old), and the padding applied 
    on the width and height as a tuple (dw, dh).

    Note:
    - The function assumes that the input image is already on the GPU.
    - The resizing operation is performed using linear interpolation.
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.size()  # current shape [width, height] for GpuMat
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) - swapping width and height to match GpuMat.size() order
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # Compute padding [width, height]
    new_unpad = (int(round(shape[0] * r)), int(round(shape[1] * r)))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape != new_unpad:  # resize
        im = cv2.cuda.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = pad_image_on_gpu_with_pytorch(im, )

    # # Since there's no direct cuda equivalent for copyMakeBorder, we need to copy the GPU mat to CPU, apply borders, and then copy back to GPU
    # im_cpu = im.download()  # Download the GpuMat to CPU for border application
    # im_cpu = cv2.copyMakeBorder(im_cpu, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # # Upload the image back to GPU
    # im = cv2.cuda_GpuMat()
    # im.upload(im_cpu)

    return im, r, (dw, dh)