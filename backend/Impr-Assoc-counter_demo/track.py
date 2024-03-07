# Import necessary libraries and modules for video processing, object detection, and tracking

import argparse
import os
from pickle import OBJ
import sys
import os.path as osp
import cv2
import numpy as np
import torch
import tensorflow as tf
import json


from YOLOv8_TensorRT import TRTModule  # isort:skip
from YOLOv8_TensorRT.torch_utils import det_postprocess
from YOLOv8_TensorRT.utils import letterbox, blob, path_to_list

sys.path.append('.')

from loguru import logger

import supervision as sv
from supervision import VideoSink, VideoInfo, get_video_frames_generator
from typing import Callable, Generator, Optional, Tuple, Iterable


# from yolox.data.data_augment import preproc
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess
# from yolox.utils.visualize import plot_tracking

# from tracker.tracking_utils.timer import Timer
from Impr_Assoc_Track.Impr_Assoc_Track import ImprAssocTrack
from ConfTrack.ConfTrack import ConfTrack
from LSTMTrack.LSTMTrack import LSTM_Track
from LSTMTrack.LSTM_predictor import LSTM_predictor
from supervision import ByteTrack
import super_gradients as sg

from ultralytics import YOLO, NAS

import color_transfer_cpu as ct_cpu
import color_transfer_gpu as ct_gpu

import csv
import pandas as pd

# Import necessary libraries and modules for video processing, object detection, and tracking
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Define the argument parser for command-line interface
def make_parser():
  parser = argparse.ArgumentParser("Track and Count People with Improved Association Track!")

  parser.add_argument("--source_video_path", help="path to source video to perform counting on. 'path/to/video.ext' ext can be: ('mp4', 'm4v', 'mjpeg', 'avi', 'h264')")
  parser.add_argument("--output_dir", help="path to target output directory. 'path/to/output/dir'")

  #     parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
  #     parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
  parser.add_argument("-f", "--count_lines_file", default=None, type=str, help="input your count lines filepath (format specified in docs)")
  parser.add_argument("-c", "--ckpt", default=None, type=str, help="path to yolo weights file")
  parser.add_argument("-expn", "--experiment-name", type=str, default=None)
  parser.add_argument("--default-parameters", dest="default_parameters", default=True, action="store_true", help="use the default parameters as in the paper")
  parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

  # Detector
  parser.add_argument("--yolo_version", default="yolov8", type=str, help="yolo model architecture. Can be 'yolov8' or 'yolo-nas'")
  parser.add_argument("--device", default="cuda", type=str, help="device to run our model, can either be cpu or cuda")
  parser.add_argument("--conf", default=None, type=float, help="test conf")
  parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
  parser.add_argument("--tsize", default=None, type=int, help="test img size")
  parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
  parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

  # tracking args  #Add in Ultralytics trackers
  parser.add_argument("--object_tracker", type=str, default="Impr_Assoc", help="which object tracker to use: 'Impr_Assoc' | 'ConfTrack' | 'LSTMTrack' | 'BYTETrack'")
  parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
  parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
  parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
  parser.add_argument("--tent_conf_thresh", default=0.7, type=float, help="threshold to start a tentative track (ConfTrack)")
  parser.add_argument("--track_buffer", type=int, default=35, help="the frames for keep lost tracks")
  parser.add_argument("--match_thresh", type=float, default=0.65, help="high confidence matching threshold for tracking")
  parser.add_argument("--second_match_thresh", type=float, default=0.19, help="low confidence matching threshold for tracking")
  parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
  parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
  parser.add_argument('--overlap_thresh', type=float, default=0.55, help='max box overlap allowed between new tracks and existing tracks')
  parser.add_argument('--iou_weight', type=float, default=0.2, help='weight of bounding box distance function in relation to ReID feature distance function')


  #     CMC
  #     parser.add_argument("--cmc-method", default="file", type=str, help="camera motion compensation method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

  # ReID
  parser.add_argument("--with_reid", dest="with_reid", action="store_true", help="use Re-ID flag.")
  parser.add_argument("--fast_reid_config", type=str, default="/content/drive/MyDrive/BWCT-tracker/Impr-Assoc-counter_demo/fast_reid/configs/MOT17/sbs_S50.yml", help="reid config file path")
  parser.add_argument("--fast_reid_weights", type=str, default=r"/content/drive/MyDrive/BWCT-tracker/Impr-Assoc-counter_demo/models/mot17_sbs_S50.pth", help="reid config file path")
  parser.add_argument('--proximity_thresh', type=float, default=0.1, help='threshold for rejecting low overlap reid matches')
  parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

  # Color Calibration
  parser.add_argument('--color_calib_enable', dest="color_calib_enable", action="store_true", help='Enable color calibration')
  parser.add_argument("--color_source_path", help="path to color source image for color correction. 'path/to/image.ext' ext must be: ('jpg')")
  parser.add_argument('--color_calib_device', type=str, default="cpu", help='which device to use for color calibration. GPU requires OpeCV with CUDA')

  parser.add_argument('--day_night_switch_file', type=str, default="", help='The path to the file that defines which camera is being used. example in "example_day_night.txt"')

  return parser

# Helper functions for creating output directories, saving configurations, and parsing files
def create_run_folder(output_dir_name):
    # Create the main output directory if it doesn't exist
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    # Count existing run folders
    existing_runs = [d for d in os.listdir(output_dir_name) if os.path.isdir(os.path.join(output_dir_name, d)) and d.startswith("run_")]
    next_run_index = len(existing_runs) + 1

    # Create new run folder
    new_run_folder = os.path.join(output_dir_name, f'run_{next_run_index}')
    os.makedirs(new_run_folder)

    return new_run_folder

def save_config(args, file_path):
    # Convert the args namespace to a dictionary
    config_dict = vars(args)

    # Write the dictionary to a file in JSON format
    with open(file_path, 'w') as file:
        json.dump(config_dict, file, indent=4)

def parse_count_lines_file(count_lines_file):
  '''File format:
  (x1,y1) (x2,y2)
  (x1,y1) (x2,y2)'''
  with open(count_lines_file, 'r') as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  lines = [line.split(' ') for line in lines]
  logger.info(f"lines pre replace: {lines}")
  lines = [[eval(coord.replace('(', '').replace(')', '')) for coord in line] for line in lines]
  return lines

def parse_day_night_file(day_night_file):
  '''File format:
  frame_id,night_true
  0,True
  15,False
  ...

  Returns dict: {frame_id: night_true}
  '''
  # TODO: define file format and parse file
  if os.path.exists(day_night_file):
    df = pd.read_csv(day_night_file)
    print(df)
    day_night_dict = df.set_index('frame_id').T.to_dict('list')
    print(day_night_dict)
    return day_night_dict
  return {}

# Function to process a video file by applying a callback function on each frame
# and saving the result to a target video file
def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    codec: str = "mp4v",
    save_video: bool = True
) -> None:
    """
    Process a video file by applying a callback function on each frame
        and saving the result to a target video file.

    Args:
        source_path (str): The path to the source video file.
        target_path (str): The path to the target video file.
        callback (Callable[[np.ndarray, int], np.ndarray]): A function that takes in
            a numpy ndarray representation of a video frame and an
            int index of the frame and returns a processed numpy ndarray
            representation of the frame.
        codec (str): The codec to write the target video to. Default 'mp4v'
        save_video (bool): Whether to write the video or not. Default True.

    Examples:
        ```python
        import supervision as sv

        def callback(scene: np.ndarray, index: int) -> np.ndarray:
            ...

        process_video(
            source_path='...',
            target_path='...',
            callback=callback
        )
        ```
    """
    source_video_info = VideoInfo.from_video_path(video_path=source_path)
    if save_video:
      with VideoSink(target_path=target_path, video_info=source_video_info, codec=codec) as sink:
          for index, frame in enumerate(
              get_video_frames_generator(source_path=source_path)
          ):
              result_frame = callback(frame, index)
              sink.write_frame(frame=result_frame)
    else:
      for index, frame in enumerate(
        get_video_frames_generator(source_path=source_path)
      ):
          callback(frame, index)
	 
if __name__ == "__main__":
  args = make_parser().parse_args()

  SOURCE_VIDEO_PATH = args.source_video_path
  SOURCE_VIDEO_NAME = osp.basename(SOURCE_VIDEO_PATH).split('.')[0]
  SOURCE_VIDEO_EXT = osp.basename(SOURCE_VIDEO_PATH).split('.')[1]
  if SOURCE_VIDEO_EXT == "avi":
    SOURCE_VIDEO_CODEC = "MJPG"
  elif SOURCE_VIDEO_EXT == "mp4":
    SOURCE_VIDEO_CODEC = "mp4v"
  OUTPUT_DIR = args.output_dir # Workspace/analysis/analysis-outputs/
  THIS_RUN_FOLDER = create_run_folder(f"{OUTPUT_DIR}/{SOURCE_VIDEO_NAME}")
  TARGET_VIDEO_PATH_ANN = f"{THIS_RUN_FOLDER}/{SOURCE_VIDEO_NAME}_annotated.{SOURCE_VIDEO_EXT}"
  TARGET_VIDEO_PATH_CLEAN = f"{THIS_RUN_FOLDER}/{SOURCE_VIDEO_NAME}_recolored.{SOURCE_VIDEO_EXT}"; print(TARGET_VIDEO_PATH_CLEAN)
  TRACK_OUTPUT_FILE_PATH = f"{THIS_RUN_FOLDER}/{SOURCE_VIDEO_NAME}_tracks_output.txt"
  # Check if the track file already exists and delete it if it does
  if os.path.exists(TRACK_OUTPUT_FILE_PATH):
      os.remove(TRACK_OUTPUT_FILE_PATH)
  COUNT_OUTPUT_FILE_PATH = f"{THIS_RUN_FOLDER}/{SOURCE_VIDEO_NAME}_counts_output.txt"
  # save configuration to config.json
  CONFIG_FILE_PATH = f"{THIS_RUN_FOLDER}/config.json"
  save_config(args, CONFIG_FILE_PATH)

  # get the color_source_path
  COLOR_SOURCE_PATH = args.color_source_path

  # get the day_night_path
  DAY_NIGHT_PATH = args.day_night_switch_file

  # get the count lines file
  COUNT_LINES_FILE = args.count_lines_file

  # Which object tracker to use
  OBJECT_TRACKER = args.object_tracker

  # save the line crossings file
  LINE_CROSSINGS_FILE = f"{THIS_RUN_FOLDER}/{SOURCE_VIDEO_NAME}_line_crossings.txt"
 
  print(f"color calib enable: {args.color_calib_enable}")
  MODEL = args.ckpt

  MODEL_EXTENSION = osp.basename(MODEL).split('.')[1]

  if args.yolo_version == 'yolov8':
    ''' load YOLOv8'''
    # change this

    # load YOLOv8
    logger.info("loading YOLOv8 model from: {}", MODEL)
    if MODEL_EXTENSION == "pt":
      yolo_model = YOLO(MODEL)
      yolo_model.fuse()
      # dict maping class_id to class_name
      CLASS_NAMES_DICT = yolo_model.model.names

    elif MODEL_EXTENSION == "engine":
      #engine = "yolov8s.engine"
      global Engine
      global device
      global H, W
      # Load a model from an .engine file
      device = torch.device(args.device)
      Engine = TRTModule(MODEL, device)
      H, W = Engine.inp_info[0].shape[-2:]
      print(H, W)
      Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

      # yolo_model = YOLO(MODEL, task="detect")
      CLASS_NAMES_DICT = {1:"Pedestrians", 0:"Bikes", 2:"Scooters", 3:"Wheelchairs"} # for yolov8n-2023-11-03 change for yolov8s-2024-02-14
      #yolo_model = torch.hub.load(MODEL, 'custom',
    else:
      yolo_model = YOLO(MODEL)
      # yolo_model.fuse()
      # dict maping class_id to class_name
      CLASS_NAMES_DICT = {0:'0', 1:'1', 2:'2', 3:'3'}

      # CLASS_NAMES_DICT = yolo_model.model.names
  elif args.yolo_version == 'yolo-nas':
    ''' load YOLO-NAS'''
    # change this

    # load YOLO-NAS
    logger.info("loading YOLO-NAS model from: {}", MODEL)

    yolo_model = sg.training.models.get(
        "yolo-nas-s", # model name
        num_classes=4, # number of classes
        checkpoint_path=MODEL, # path to the checkpoint
    ).to(torch.device("cuda:0"))
    CLASS_NAMES_DICT = {0:'0', 1:'1', 2:'2', 3:'3'}

    # CLASS_NAMES_DICT = yolo_model.model.names


  # class_ids of interest - pedestrians, bikes, scooters, wheelchairs
  selected_classes = [0,1,2,3]
  print(CLASS_NAMES_DICT)

  video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

  # parse the count lines file
  count_lines = parse_count_lines_file(COUNT_LINES_FILE)
  line_zones = []
  with open(LINE_CROSSINGS_FILE, 'w') as f:
    # Change to use BOTTOM_CENTER as the trigger for a count (more intuitive)
    # triggering_anchors = [sv.Position.BOTTOM_CENTER]
    # Try triggering on bottom left and bottom right
    triggering_anchors = [sv.Position.BOTTOM_LEFT, sv.Position.BOTTOM_RIGHT]

    for i, line in enumerate(count_lines):
      logger.info(f"line {line}")
      # add all of the line zone counters to a list
      line_zones.append(sv.LineZone(start=sv.Point(line[0][0], line[0][1]), end=sv.Point(line[1][0], line[1][1]), triggering_anchors=triggering_anchors))
      f.write(f"line_{i}: ({line[0]}, {line[1]}),")
    f.write("\n")
  print(f"line_zones {line_zones}")

  # parse the day night file
  camera_switches_dict = {}
  if DAY_NIGHT_PATH != "":
    # dict of frame id's to which camera we are switching to.
    camera_switches_dict = parse_day_night_file(DAY_NIGHT_PATH)
    color_calib_enable = True
  elif args.color_calib_enable:
    color_calib_enable = True
  else:
    color_calib_enable = False

  print(f"Color calibration on? {color_calib_enable}")

# Initialize the object tracker based on the specified tracker type
  if OBJECT_TRACKER == "Impr_Assoc":
    if args.default_parameters:
      Tracker = ImprAssocTrack(with_reid=args.with_reid,
                              fast_reid_config=args.fast_reid_config,
                              fast_reid_weights=args.fast_reid_weights,
                              device=args.device,
                              frame_rate=video_info.fps
                              )
    else:
      Tracker = ImprAssocTrack(track_high_thresh=args.track_high_thresh,
                                      track_low_thresh=args.track_low_thresh,
                                      new_track_thresh=args.new_track_thresh,
                                      track_buffer=args.track_buffer,
                                      match_thresh=args.match_thresh,
                                      second_match_thresh=args.second_match_thresh,
                                      overlap_thresh=args.overlap_thresh,
                                      iou_weight=args.iou_weight,
                                      proximity_thresh=args.proximity_thresh,
                                      appearance_thresh=args.appearance_thresh,
                                      with_reid=args.with_reid,
                                      fast_reid_config=args.fast_reid_config,
                                      fast_reid_weights=args.fast_reid_weights,
                                      device=args.device,
                                      frame_rate=video_info.fps)
  elif OBJECT_TRACKER == "ConfTrack":
    if args.default_parameters:
      print(f"with reid {args.with_reid}")
      Tracker = ConfTrack(with_reid=args.with_reid,
                          fast_reid_config=args.fast_reid_config, #need to download
                          fast_reid_weights=args.fast_reid_weights, #need to download
                          device=args.device,
                          frame_rate=video_info.fps)
    else:
      Tracker = ConfTrack(track_high_thresh=args.track_high_thresh,
                          track_low_thresh=args.track_low_thresh,
                          new_track_thresh=args.new_track_thresh,
                          tent_conf_thresh=args.tent_conf_thresh,
                          match_thresh=args.match_thresh,
                          track_buffer=args.track_buffer,
                          proximity_thresh=args.proximity_thresh,
                          appearance_thresh=args.appearance_thresh,
                          with_reid=args.with_reid,
                          fast_reid_config=args.fast_reid_config, #need to download
                          fast_reid_weights=args.fast_reid_weights, #need to download
                          device=args.device,
                          frame_rate=video_info.fps)
  elif OBJECT_TRACKER == "LSTMTrack":
    from LSTMTrack.LSTMTrack import STrack
    # currently load model 14 and then load model 15 bbox weights, need to fix
    LSTM_model = tf.keras.models.load_model("/home/object_track_count_analysis/BWCT-tracker/Impr-Assoc-counter_demo/models/LSTM_model_14_and_15_bb")
    # LSTM_model.load_weights("./models/model_15_bb_saved_weights.h5")
    if args.default_parameters:
      STrack.shared_LSTM_predictor = LSTM_predictor(LSTM_model)
      Tracker = LSTM_Track(model=LSTM_model,
                          torchreid_model=r"/home/object_track_count_analysis/BWCT-tracker/Impr-Assoc-counter_demo/models/osnet_ms_d_c.pth.tar", # need to move to folder
                          frame_rate=video_info.fps)
    else:
      STrack.shared_LSTM_predictor = LSTM_predictor(LSTM_model)
      # add in params
      Tracker = LSTM_Track(model=LSTM_model,
                          track_thresh=args.track_low_thresh,
                          track_buffer=args.track_buffer,
                          match_thresh=args.track_match_thresh,
                          torchreid_model=r"/home/object_track_count_analysis/BWCT-tracker/Impr-Assoc-counter_demo/models/osnet_ms_d_c.pth.tar", # need to move to folder
                          frame_rate=video_info.fps)
  elif OBJECT_TRACKER == "BYTETrack":
    if args.default_parameters:
      Tracker = ByteTrack(track_thresh=0.25,
                          track_buffer=20,
                          match_thresh=0.8,
                          frame_rate=video_info.fps)
    else:
      Tracker = ByteTrack(track_thresh=args.track_low_thresh,
                          track_buffer=args.track_buffer,
                          match_thresh=args.track_match_thresh,
                          frame_rate=video_info.fps)
   

  # create instance of BoxAnnotator
  box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=2, text_scale=1)

  # create instance of TraceAnnotator
  trace_annotator = sv.TraceAnnotator(thickness=1, trace_length=300)

  # create instance of LineZoneAnnotator
  line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

  # create instance of FPSMonitor
  fps_monitor = sv.FPSMonitor()

# Check if color calibration is enabled
  if color_calib_enable:
    # Check if frames should be saved
    if args.save_frames:
      # Check the video extension and create a VideoWriter object accordingly
      if SOURCE_VIDEO_EXT == 'avi':
        out = cv2.VideoWriter(TARGET_VIDEO_PATH_CLEAN, cv2.VideoWriter_fourcc(*'MJPG'), video_info.fps, (video_info.width, video_info.height))
      else:
        out = cv2.VideoWriter(TARGET_VIDEO_PATH_CLEAN, cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps, (video_info.width, video_info.height))
    # Check the device for color calibration
    if args.color_calib_device=="cpu":
      '''for cpu color correction'''
      # Read the color source image and convert it to LAB color space
      color_source = cv2.imread(COLOR_SOURCE_PATH)
      color_source = cv2.cvtColor(color_source, cv2.COLOR_BGR2LAB).astype(np.float32)
      # Calculate the statistics of the source image
      source_img_stats = ct_cpu.image_stats(color_source)
      print(source_img_stats)

    elif args.color_calib_device=="gpu":
      '''for gpu version'''
      # Read the color source image and resize it to match the video dimensions
      color_source = cv2.imread(COLOR_SOURCE_PATH)
      color_source = cv2.resize(color_source, (video_info.width, video_info.height), interpolation=cv2.INTER_LINEAR)
      # Upload the color source image to the GPU
      gpu_color_source = cv2.cuda_GpuMat()
      gpu_color_source.upload(color_source)
      # Convert the color source image to LAB color space on the GPU
      cv2.cuda.cvtColor(gpu_color_source, cv2.COLOR_BGR2LAB, gpu_color_source)
      # Calculate the statistics of the source image on the GPU
      source_img_stats = ct_gpu.image_stats_gpu(gpu_color_source)

      # Create a GpuMat object for the frame
      gpu_frame = cv2.cuda_GpuMat()
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

  total_fps = 0

  def callback(frame: np.ndarray, frame_id: int, color_calib_device='cpu') -> np.ndarray:
    """
    This callback function is invoked for each frame of a video during the processing pipeline
    implemented in Roboflow's `process_video` function. It performs a series of operations on the
    given frame, including optional color calibration, object detection, object tracking, annotating,
    and logging. It supports conditional color calibration based on a global flag and device specification,
    utilizes different models for object detection, tracks objects across frames, annotates frames with
    detection and tracking information, and logs progress and line crossing events.

    Parameters:
    - frame (np.ndarray): The current video frame to be processed. It is a NumPy array representing
      the image data in BGR.
    - frame_id (int): The identifier of the current frame. It is used for logging, tracking progress,
      and conditional operations based on the frame sequence.
    - color_calib_device (str, optional): Specifies the device to be used for color calibration. The default
      is 'cpu', but it can be set to 'gpu'.

    Returns:
    - np.ndarray: The processed frame, potentially color-calibrated, annotated with object detections,
      tracking information, and other annotations depending on the pipeline configuration.

    The function internally manages several global variables for state tracking, including color calibration
    flags, output video stream configuration, performance monitoring, and detection/tracking configurations.
    It adapts to different configurations and models dynamically, supporting a flexible video processing
    pipeline. This function is designed to be used as part of a larger video processing workflow, where
    it is passed as a callback to a video processing utility function, allowing for custom processing logic
    on a per-frame basis.
    """
    global source_img_stats, out, fps_monitor, line_counts, args, total_fps, total_frames, color_calib_enable, camera_switches_dict
    # Check if the current frame ID is in the dictionary of camera switches
    if frame_id in camera_switches_dict.keys():
        # If the value for this frame ID is True, enable color calibration
        if camera_switches_dict[frame_id] == [True]:
            print(f"Switch to CC on! Frame: {frame_id}")
            color_calib_enable = True
        # If the value for this frame ID is False, disable color calibration
        elif camera_switches_dict[frame_id] == [False]:
            print(f"Switch to CC off! Frame: {frame_id}")
            color_calib_enable = False

    # If color calibration is enabled
    if color_calib_enable:
        ''' Color Calibration '''
        # If the color calibration device is set to 'cpu', perform color transfer on the CPU
        if args.color_calib_device == 'cpu':
            frame_cpu = ct_cpu.color_transfer_cpu(source_img_stats, frame, clip=False, preserve_paper=False)
        # If the color calibration device is set to 'gpu', perform color transfer on the GPU
        elif args.color_calib_device == 'gpu':
            gpu_frame.upload(frame)
            frame_gpuMat = ct_gpu.color_transfer_gpu(source_img_stats, gpu_frame, clip=False, preserve_paper=False)
            frame_cpu = frame_gpuMat.download()
            frame = ct_gpu.gpu_mat_to_torch_tensor(frame_gpuMat)
        # If the save_frames argument is set, write the frame to the output video
        if args.save_frames:
            out.write(frame_cpu)
    else:
        frame_cpu = frame

    ''' Detection '''
    # If the YOLO version is set to 'yolov8'
    if args.yolo_version == 'yolov8':
        # If the model extension is 'engine'
        if MODEL_EXTENSION == "engine":
            # Resize and pad the frame to the desired size (W, H) while keeping the aspect ratio
            # The function also returns the scaling ratio and the width and height padding
            bgr, ratio, dwdh = letterbox(frame_cpu, (W,H))
            
            # Convert the color space from BGR to RGB as the model was trained on RGB images
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # Convert the image to a tensor and normalize it to the range [0, 1]
            tensor = blob(rgb, return_seg=False)
            
            # Convert the width and height padding to a tensor and double its value
            # This is done because the padding was divided by 2 when it was added to both sides of the image
            dwdh = torch.asarray(dwdh*2, dtype=torch.float32, device=device)
            
            # Move the image tensor to the specified device (CPU or GPU)
            tensor = torch.asarray(tensor, device=device)
            
            # Run the image tensor through the model and get the output
            data = Engine(tensor)
            
            # Post-process the model output to get the bounding boxes, scores, and labels
            bboxes, scores, labels = det_postprocess(data)
            
            # Create an empty Detections object to store the detection results
            detections = sv.Detections.empty()
            
            # Convert the bounding boxes from the model output space to the original image space
            # This is done by subtracting the padding and dividing by the scaling ratio
            detections.xyxy = ((bboxes-dwdh)/ratio).cpu().numpy()
            
            # Store the scores and labels in the Detections object
            detections.confidence = scores.cpu().numpy()
            detections.class_id = labels.cpu().numpy().astype(int)
        else:
            results = yolo_model.predict(frame_cpu, verbose=False, iou=0.7, conf=0.1, device="cuda")[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[np.isin(detections.class_id, selected_classes)]
    # If the YOLO version is set to 'yolo-nas'
    elif args.yolo_version == 'yolo-nas':
        results = yolo_model.predict(frame_cpu, iou=0.7, conf=0.1, fuse_model=False)
        detections = sv.Detections.from_yolo_nas(results)
        detections = detections[np.isin(detections.class_id, selected_classes)]
    ''' Tracking '''
    # commented one for impr_associated
    if OBJECT_TRACKER == "BYTETrack":
      detections = Tracker.update_with_detections(detections)
    else:
      detections = Tracker.update_with_detections(detections, frame_cpu)
    # detections = impr_assoc_tracker.update_with_detections(detections)

    ''' Save Tracks '''
    with open(TRACK_OUTPUT_FILE_PATH, 'a+', newline='', encoding='UTF8') as f:
      writer = csv.writer(f)
      for track, _, conf, class_id, tracker_id, _ in detections:
        writer.writerow([frame_id, tracker_id, track[0], track[1], track[2]-track[0], track[3]-track[1], conf, class_id, -1, -1])

    ''' Annotate '''
    labels = [
      f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
      for _, _, confidence, class_id, tracker_id, _
      in detections
    ]
    if detections.tracker_id.size is None:
      annotated_frame = frame_cpu.copy()
    else:
      annotated_frame = trace_annotator.annotate(scene=frame_cpu.copy(), detections=detections)
 
    annotated_frame=box_annotator.annotate(
      scene=annotated_frame,
      detections=detections,
      labels=labels)
    for line_zone in line_zones:
      annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)


    ''' Update line counter '''
    # Need to write to a file that stores the counts in this format:
    # <frame_id>,<line number>,<class name>,<in/out>
    for i, line_zone in enumerate(line_zones):
        objects_in, objects_out = line_zone.trigger(detections)
        for obj in detections.class_id[np.isin(objects_in, True)]:
            line_counts[i][CLASS_NAMES_DICT[obj]+"_in"] += 1
            with open(LINE_CROSSINGS_FILE, 'a') as f:
                f.write(f"{frame_id},{i},{CLASS_NAMES_DICT[obj]},in\n")
        for obj in detections.class_id[np.isin(objects_out, True)]:
            line_counts[i][CLASS_NAMES_DICT[obj]+"_out"] += 1
            with open(LINE_CROSSINGS_FILE, 'a') as f:
                f.write(f"{frame_id},{i},{CLASS_NAMES_DICT[obj]},out\n")

    fps_monitor.tick()
    total_fps += fps_monitor()
    total_frames = frame_id + 1
    ''' Log Time'''
    if frame_id % 20 == 0:
      # logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, video_info.total_frames, max(1e-5, total_fps/total_frames)))
      progress = frame_id / video_info.total_frames * 100  # Calculate progress as a percentage
      with open('progress.txt', 'w') as f:
        f.write(str(progress))
      with open('track_logs.txt', '+a') as f:
        f.write('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, video_info.total_frames, max(1e-5, fps_monitor())))
      # logger.info(f"Write Progress: {progress}")

    # ''' Log Time'''
    # if frame_id % 20 == 0:
    #     progress = frame_id / video_info.total_frames * 100  # Calculate progress as a percentage
    #     yield progress  # Yield progress updates to the client
      
    return annotated_frame

  ''' Now process the whole video '''
  logger.info(f"saving results to {TRACK_OUTPUT_FILE_PATH}")
  logger.info(f"Save frames: {args.save_frames}")
  process_video( 
    source_path = SOURCE_VIDEO_PATH, 
    target_path = TARGET_VIDEO_PATH_ANN,
    callback=callback,
    # codec=SOURCE_VIDEO_CODEC
    codec="mp4v",
    save_video=args.save_frames
  )
if args.color_calib_enable:
  out.release() 
 

''' Save Counts ''' 
with open(COUNT_OUTPUT_FILE_PATH, 'a+', newline='', encoding='UTF8') as f:
  writer = csv.writer(f)
  for i, line_count in enumerate(line_counts):
    writer.writerow([f"line {i}"])
    for key, val in line_count.items():
      writer.writerow([key, val])
  writer.writerow([f"Average FPS: {total_fps/total_frames}"])

