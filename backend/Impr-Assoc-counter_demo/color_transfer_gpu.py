import cv2
import numpy as np
import cupy as cp

import torch


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

def create_scalar_gpumat(value, size, type):
    """Create a cv2.cuda_GpuMat with the given size and type, filled with the specified value."""
    # Create a NumPy array filled with the scalar value
    scalar_array = np.full((1, 1), value, dtype=np.float32)
    # Create a GpuMat and upload the NumPy array
    mat = cv2.cuda_GpuMat()
    mat.upload(scalar_array)

    # print(f"size: {size}")
    # Resize the GpuMat to the desired size
    mat_resized = cv2.cuda.GpuMat((size[0], size[1]), type)
    cv2.cuda.resize(mat, (size[0], size[1]), mat_resized)  # Note: OpenCV uses width x height
    # mat.release()
    # print(f"resized: {mat_resized.download()}")
    return mat_resized

def cv_cuda_gpumat_from_cp_array(cupy_array: cp.ndarray) -> cv2.cuda.GpuMat:
    h, w = cupy_array.shape[:2]
    cuda_array_interface = cupy_array.__cuda_array_interface__
    data_ptr = int(cuda_array_interface['data'][0])
    typestr = cuda_array_interface['typestr']
    channels = 1 if len(cupy_array.shape) == 2 else cupy_array.shape[2]

    # Map the typestr to OpenCV type
    type_map = {
        '|u1': cv2.CV_8U,
        '|i1': cv2.CV_8S,
        '<u2': cv2.CV_16U,
        '<i2': cv2.CV_16S,
        '<i4': cv2.CV_32S,
        '<f4': cv2.CV_32F,
        '<f8': cv2.CV_64F,
    }
    cv_type = type_map[typestr]

    # Create a GpuMat with the same size and type
    mat = cv2.cuda_GpuMat()
    mat.create((w, h), cv_type)


    # Get the address of the CuPy array
    cupy_data_ptr = cupy_array.data.ptr

    # Use CuPy to copy data directly between arrays in GPU memory
    # Attempting a different approach by manually invoking the CUDA memcpy function
    cp.cuda.runtime.memcpy(mat.data, cupy_data_ptr, cupy_array.nbytes, cp.cuda.runtime.memcpyDeviceToDevice)

    return mat

def cupy_to_gpumat(cupy_array):
    """Convert a CuPy array to a cv2.cuda_GpuMat."""
    h, w = cupy_array.shape[:2]
    cuda_array_interface = cupy_array.__cuda_array_interface__
    data_ptr = int(cuda_array_interface['data'][0])
    typestr = cuda_array_interface['typestr']
    channels = 1 if len(cupy_array.shape) == 2 else cupy_array.shape[2]

    # Map the typestr to OpenCV type
    type_map = {
        '|u1': cv2.CV_8U,
        '|i1': cv2.CV_8S,
        '<u2': cv2.CV_16U,
        '<i2': cv2.CV_16S,
        '<f4': cv2.CV_32F,
        '<f8': cv2.CV_64F,
    }
    cv_type = type_map[typestr]

    # Create a GpuMat and set its data pointer
    gpumat = cv2.cuda_GpuMat(h, w, cv_type, data_ptr)
    return gpumat

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

"""Create a cv2.cuda_GpuMat with the given size and type, filled with the specified value."""
# Create a NumPy array filled with the scalar value
scalar_array = np.full((1, 1, 3), 255, dtype=np.float32)
# Create a GpuMat and upload the NumPy array
mat = cv2.cuda_GpuMat()
mat.upload(scalar_array)
# print(mat.download())

# Resize the GpuMat to the desired size
all_white_img_gpu = cv2.cuda.GpuMat((1280, 720), cv2.CV_32FC3)
cv2.cuda.resize(mat, (1280, 720), all_white_img_gpu)  # Note: OpenCV uses width x height
# mat.release()
# print(f"resized: {all_white_img_gpu.download()}")
# all_white_img_gpu = create_scalar_gpumat(255, (1280, 720), cv2.CV_32FC3)

def color_transfer_gpu(source_img_stats, target_gpu, clip=True, preserve_paper=True):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space, with all processing done on the GPU.
    """
    global all_white_img_gpu
    # Convert the target image from the BGR to L*a*b* color space (GPU)
    # print(f"target in bgr initial: {target_gpu.download()} type: {target_gpu.type()}")

    target_lab_gpu = cv2.cuda.cvtColor(target_gpu, cv2.COLOR_BGR2Lab)
    target_gpu_size = target_gpu.size()
    target_gpu.release()
    # print(f"target in lab initial: {target_lab_gpu.download()} type: {target_lab_gpu.type()}")

    target_lab_gpu_32f = cv2.cuda.GpuMat(target_lab_gpu.size(), cv2.CV_32FC3)
    target_lab_gpu.convertTo(cv2.CV_32F, target_lab_gpu_32f)
    target_lab_gpu.release()
    # print(f"target in lab 32f initial: {target_lab_gpu_32f.download()} type: {target_lab_gpu_32f.type()}")


    # Compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = source_img_stats
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats_gpu(target_lab_gpu_32f)

    # print(f"img stats: {lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc}")

    # Split the channels (GPU)
    l_gpu, a_gpu, b_gpu = cv2.cuda.split(target_lab_gpu_32f)

    l_gpu_32f = cv2.cuda.GpuMat(l_gpu.size(), cv2.CV_32F)
    a_gpu_32f = cv2.cuda.GpuMat(a_gpu.size(), cv2.CV_32F)
    b_gpu_32f = cv2.cuda.GpuMat(b_gpu.size(), cv2.CV_32F)

    # print(f"l_gpu: {l_gpu.download()}")

    # Ensure LAB channels are of type CV_32F
    l_gpu.convertTo(cv2.CV_32F, l_gpu_32f)
    a_gpu.convertTo(cv2.CV_32F, a_gpu_32f)
    b_gpu.convertTo(cv2.CV_32F, b_gpu_32f)
    l_gpu.release()
    a_gpu.release()
    b_gpu.release()

    # print(f"l_gpu_32f: {l_gpu_32f.download()}")


    # print("l_gpu size:", l_gpu.size(), "type:", l_gpu.type())
    # print("l_gpu_32f size:", l_gpu_32f.size(), "type:", l_gpu_32f.type())

    # print("lMeanTar size:", lMeanTar.size(), "type:", lMeanTar.type())
    # print("lStdSrc size:", lStdSrc.size(), "type:", lStdSrc.type())

    lMeanTar_gpu = create_scalar_gpumat(lMeanTar, l_gpu_32f.size(), cv2.CV_32F)
    aMeanTar_gpu = create_scalar_gpumat(aMeanTar, a_gpu_32f.size(), cv2.CV_32F)
    bMeanTar_gpu = create_scalar_gpumat(bMeanTar, b_gpu_32f.size(), cv2.CV_32F)

    lMeanSrc_gpu = create_scalar_gpumat(lMeanSrc, l_gpu_32f.size(), cv2.CV_32F)
    aMeanSrc_gpu = create_scalar_gpumat(aMeanSrc, a_gpu_32f.size(), cv2.CV_32F)
    bMeanSrc_gpu = create_scalar_gpumat(bMeanSrc, b_gpu_32f.size(), cv2.CV_32F)

    # Subtract the means from the target image (GPU)
    cv2.cuda.subtract(l_gpu_32f, lMeanTar_gpu, l_gpu_32f)
    cv2.cuda.subtract(a_gpu_32f, aMeanTar_gpu, a_gpu_32f)
    cv2.cuda.subtract(b_gpu_32f, bMeanTar_gpu, b_gpu_32f)

    lMeanTar_gpu.release()
    aMeanTar_gpu.release()
    bMeanTar_gpu.release()

    # print(f"l_gpu_32f - lMean: {l_gpu_32f.download()}")


    # l_scaler_gpu = cv2.cuda.GpuMat(l_gpu.size(), cv2.CV_32F)
    # a_scaler_gpu = cv2.cuda.GpuMat(a_gpu.size(), cv2.CV_32F)
    # b_scaler_gpu = cv2.cuda.GpuMat(b_gpu.size(), cv2.CV_32F)

    # Scale by the standard deviations (GPU)
    if preserve_paper:

        l_ratio = lStdTar / lStdSrc
        # print(f"ratio: {lStdTar / lStdSrc}")

        a_ratio = aStdTar / aStdSrc
        b_ratio = bStdTar / bStdSrc

        l_scaler_gpu = create_scalar_gpumat(l_ratio, l_gpu_32f.size(), cv2.CV_32F)
        a_scaler_gpu = create_scalar_gpumat(a_ratio, a_gpu_32f.size(), cv2.CV_32F)
        b_scaler_gpu = create_scalar_gpumat(b_ratio, b_gpu_32f.size(), cv2.CV_32F)


        cv2.cuda.multiply(l_gpu_32f, l_scaler_gpu, l_gpu_32f)
        cv2.cuda.multiply(a_gpu_32f, a_scaler_gpu, a_gpu_32f)
        cv2.cuda.multiply(b_gpu_32f, b_scaler_gpu, b_gpu_32f)
    else:
        # cv2.cuda.divide(lStdSrc, lStdTar, l_scaler_gpu)
        # cv2.cuda.divide(aStdSrc, aStdTar, a_scaler_gpu)
        # cv2.cuda.divide(bStdSrc, bStdTar, b_scaler_gpu)
        # print(f"l_gpu_ratio: {l_scaler_gpu.download()}")

        l_ratio = lStdSrc / lStdTar
        # print(f"ratio: {lStdSrc / lStdTar}")

        a_ratio = aStdSrc / aStdTar
        b_ratio = bStdSrc / bStdTar

        l_scaler_gpu = create_scalar_gpumat(l_ratio, l_gpu_32f.size(), cv2.CV_32F)
        a_scaler_gpu = create_scalar_gpumat(a_ratio, a_gpu_32f.size(), cv2.CV_32F)
        b_scaler_gpu = create_scalar_gpumat(b_ratio, b_gpu_32f.size(), cv2.CV_32F)

        cv2.cuda.multiply(l_gpu_32f, l_scaler_gpu, l_gpu_32f)
        cv2.cuda.multiply(a_gpu_32f, a_scaler_gpu, a_gpu_32f)
        cv2.cuda.multiply(b_gpu_32f, b_scaler_gpu, b_gpu_32f)

    # Add in the source mean (GPU)
    cv2.cuda.add(l_gpu_32f, lMeanSrc_gpu, l_gpu_32f)
    cv2.cuda.add(a_gpu_32f, aMeanSrc_gpu, a_gpu_32f)
    cv2.cuda.add(b_gpu_32f, bMeanSrc_gpu, b_gpu_32f)

    lMeanSrc_gpu.release()
    aMeanSrc_gpu.release()
    bMeanSrc_gpu.release()

    # print(f"l_gpu_32f * ratio: {l_gpu_32f.download()}")


    # Clip/scale the pixel intensities to [0, 255] if they fall outside this range (GPU)
    l_gpu_32f = scale_array_gpu(l_gpu_32f, clip=clip)
    a_gpu_32f = scale_array_gpu(a_gpu_32f, clip=clip)
    b_gpu_32f = scale_array_gpu(b_gpu_32f, clip=clip)

    # Merge the channels together and convert back to the BGR color space (GPU)
    transfer_lab_gpu = cv2.cuda.GpuMat(target_gpu_size, cv2.CV_32FC3)
    # print(f"transfer_lab_gpu shape: {transfer_lab_gpu.size()}")
    cv2.cuda.merge([l_gpu_32f, a_gpu_32f, b_gpu_32f], transfer_lab_gpu)
    # l_gpu_32f.release()
    # a_gpu_32f.release()
    # b_gpu_32f.release()

    # print(f"image in lab [0,255]: {transfer_lab_gpu.download()}")

    transfer_bgr_gpu = cv2.cuda.GpuMat(target_gpu_size, cv2.CV_8UC3)
    transfer_lab_gpu.convertTo(cv2.CV_8UC3, transfer_bgr_gpu)

    cv2.cuda.cvtColor(transfer_bgr_gpu, cv2.COLOR_LAB2BGR, transfer_bgr_gpu)
    # print(f"image in bgr [0,1]: {transfer_lab_gpu.download()}")

    # cv2.cuda.multiply(transfer_lab_gpu, all_white_img_gpu, transfer_lab_gpu)
    # print(f"image in bgr [0, 255]: {transfer_bgr_gpu.download()}")

    # Return the color transferred image
    # print(f"transfer bgr gpu: {transfer_bgr_gpu.size()}")
    return transfer_bgr_gpu

def image_stats_gpu(image_gpu):
    """
    Compute the mean and standard deviation of each channel in the L*a*b* color space.
    All computations are done on the GPU.
    """
    channels = cv2.cuda.split(image_gpu)
    stats = []


    for chan in channels:
        # Assuming you have a cv2.cuda_GpuMat object named gpu_mat
        # print(f"chan: {chan.download()} chan shape: {chan.size()}")
        # wrapper = GpuMatWrapper(chan)
        # cupy_array = cp.asarray(wrapper)
        cupy_array = cp_array_from_cv_cuda_gpumat(chan)
        # print(f"cupy_array: {cupy_array} cupy shape: {chan.size()}")
        mean = cupy_array.mean().get().astype(cp.float32)
        std_dev = cupy_array.std().get().astype(cp.float32)
        # mean = chan.mean().get()
        # std_dev = chan.std().get
        # mean_stddev = cv2.cuda.meanStdDev(chan)
        # mean_gpu = cupy_to_gpumat(mean)
        # std_dev_gpu = cupy_to_gpumat(std_dev)
        # mean_gpu = create_scalar_gpumat(mean, chan.size(), cv2.CV_32F)
        # std_dev_gpu = create_scalar_gpumat(std_dev, chan.size(), cv2.CV_32F)

        stats.extend([mean, std_dev])

    return tuple(stats)

def scale_array_gpu(arr_gpu, clip=False):
    """
    Trim GPU array values to be in [0, 255] range with option of clipping or scaling.
    """
    if clip:
        scaled_gpu = cv2.cuda.threshold(arr_gpu, 0, 255, cv2.THRESH_TRUNC)[1]
        scaled_gpu = cv2.cuda.threshold(scaled_gpu, 0, 255, cv2.THRESH_TOZERO)[1]
    else:
        # Implement min-max scaling on GPU (omitted for brevity)
        # wrapper = GpuMatWrapper(arr_gpu)
        cupy_array = cp_array_from_cv_cuda_gpumat(arr_gpu)
        min_val = cupy_array.min().get()
        max_val = cupy_array.max().get()
        # print(f"min: {min_val}, max: {max_val}")
        # min, max = cv2.cuda.minMax(arr_gpu)

        scale_range = (max([min_val, 0]), min([max_val, 255]))

        min_gpu = create_scalar_gpumat(min_val, arr_gpu.size(), cv2.CV_32F)
        scale_min_gpu = create_scalar_gpumat(scale_range[0], arr_gpu.size(), cv2.CV_32F)
        # print(f"min_gpu: {min_gpu.download()}")
        max_gpu = create_scalar_gpumat(max_val, arr_gpu.size(), cv2.CV_32F)
        # print(f"max_gpu: {max_gpu.download()}")
        scale_range_gpu = create_scalar_gpumat(scale_range[1]-scale_range[0], arr_gpu.size(), cv2.CV_32F)
        scaled_gpu = cv2.cuda.GpuMat(arr_gpu.size(), cv2.CV_32F)
        if min_val < scale_range[0] or max_val > scale_range[1]:
          # print("needs scaling")
          cv2.cuda.subtract(arr_gpu, min_gpu, scaled_gpu)
          # print(f"arr-min: {scaled_gpu.download()}")
          range_diff = create_scalar_gpumat(max_val-min_val, arr_gpu.size(), cv2.CV_32F)
          # print(f"max-min: {range_diff.download()}")
          # scale_range_gpu = create_scalar_gpumat(scale_range[1]-scale_range[0], arr_gpu.size(), cv2.CV_32F)
          cv2.cuda.divide(scaled_gpu, range_diff, scaled_gpu)
          # print(f"(arr-min)/(mx-min): {scaled_gpu.download()}")
          cv2.cuda.multiply(scale_range_gpu, scaled_gpu, scaled_gpu)
          # print(f"(new_range[1] - new_range[0]) * (arr - mn) / (mx - mn): {scaled_gpu.download()}")
          cv2.cuda.add(scaled_gpu, scale_min_gpu, scaled_gpu)
          # print(f"scaled_gpu: {scaled_gpu.download()}")
        else:
          scaled_gpu = arr_gpu
    # print(f"scaled_gpu: {scaled_gpu.download()}")
    return scaled_gpu

# Example usage
# target_gpu = cv2.cuda_GpuMat()  # Assume target image is loaded into a GpuMat
# source_img_stats = ...  # Assume source image statistics are calculated
# transfer_gpu = color_transfer_gpu(source_img_stats, target_gpu)
