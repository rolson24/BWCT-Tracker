
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
# import color_transfer_gpu as ct_gpu

import csv

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

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
  # parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

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

  return parser

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
	 
if __name__ == "__main__":
  args = make_parser().parse_args()

  SOURCE_VIDEO_PATH = args.source_video_path
  SOURCE_VIDEO_NAME = osp.basename(SOURCE_VIDEO_PATH).split('.')[0]
  OUTPUT_DIR = args.output_dir # Workspace/analysis/analysis-outputs/
  THIS_RUN_FOLDER = create_run_folder(f"{OUTPUT_DIR}/{SOURCE_VIDEO_NAME}")
  TARGET_VIDEO_PATH_ANN = f"{THIS_RUN_FOLDER}/{SOURCE_VIDEO_NAME}_annotated.mp4"
  TARGET_VIDEO_PATH_CLEAN = f"{THIS_RUN_FOLDER}/{SOURCE_VIDEO_NAME}_recolored.mp4"; print(TARGET_VIDEO_PATH_CLEAN)
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
  COUNT_LINES_FILE = args.count_lines_file
  OBJECT_TRACKER = args.object_tracker
 
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
      CLASS_NAMES_DICT = {0:'0', 1:'1', 2:'2', 3:'3'}
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

  count_lines = parse_count_lines_file(COUNT_LINES_FILE)
  line_zones = []
  for line in count_lines:
    logger.info(f"line {line}")
    line_zones.append(sv.LineZone(start=sv.Point(line[0][0], line[0][1]), end=sv.Point(line[1][0], line[1][1])))
  print(f"line_zones {line_zones}")

  # impr_assoc_tracker = ImprAssocTrack(track_high_thresh=args.track_high_thresh,
  #                                       track_low_thresh=args.track_low_thresh,
  #                                       new_track_thresh=args.new_track_thresh,
  #                                       track_buffer=args.track_buffer,
  #                                       match_thresh=args.match_thresh,
  #                                       second_match_thresh=args.second_match_thresh,
  #                                       overlap_thresh=args.overlap_thresh,
  #                                       iou_weight=args.iou_weight,
  #                                       proximity_thresh=args.proximity_thresh,
  #                                       appearance_thresh=args.appearance_thresh,
  #                                       with_reid=args.with_reid,
  #                                       fast_reid_config=args.fast_reid_config,
  #                                       fast_reid_weights=args.fast_reid_weights,
  #                                       device=args.device,
  #                                       frame_rate=video_info.fps)

  #	impr_assoc_tracker = sv.ByteTrack(track_thresh=0.25,
  #									 track_buffer=20,
  #									 match_thresh=0.8,
  #									 frame_rate=video_info.fps
  #									 )
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
    LSTM_model = tf.keras.models.load_model("/content/drive/MyDrive/BWCT-tracker/Impr-Assoc-counter_demo/models/LSTM_model_14_and_15_bbL")
    # LSTM_model.load_weights("./models/model_15_bb_saved_weights.h5")
    if args.default_parameters:
      STrack.shared_LSTM_predictor = LSTM_predictor(LSTM_model)
      Tracker = LSTMTrack(model=LSTM_model,
                          with_reid=args.with_reid,
                          torchreid_model=r"/content/drive/MyDrive/BWCT-tracker/Impr-Assoc-counter_demo/models/osnet_ms_d_c.pth.tar", # need to move to folder
                          frame_rate=video_info.fps)
    else:
      STrack.shared_LSTM_predictor = LSTM_predictor(LSTM_model)
      # add in params
      Tracker = LSTMTrack(model=LSTM_model,
                          with_reid=args.with_reid,
                          track_thresh=args.track_low_thresh,
                          track_buffer=args.track_buffer,
                          match_thresh=args.track_match_thresh,
                          torchreid_model=r"/content/drive/MyDrive/BWCT-tracker/Impr-Assoc-counter_demo/models/osnet_ms_d_c.pth.tar", # need to move to folder
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

  if args.color_calib_enable:
    out = cv2.VideoWriter(TARGET_VIDEO_PATH_CLEAN, cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps, (video_info.width, video_info.height))
    if args.color_calib_device=="cpu":
      '''for cpu color correction'''
      color_source = cv2.imread(COLOR_SOURCE_PATH)
      color_source = cv2.cvtColor(color_source, cv2.COLOR_BGR2LAB).astype(np.float32)
      source_img_stats = ct_cpu.image_stats(color_source)
      print(source_img_stats)

    elif args.color_calib_device=="gpu":
      '''for gpu version'''
      # Color source image to correct colors GPU
      color_source = cv2.imread(COLOR_SOURCE_PATH)
      color_source = cv2.resize(color_source, (video_info.width, video_info.height), interpolation=cv2.INTER_LINEAR)
      gpu_color_source = cv2.cuda_GpuMat()
      gpu_color_source.upload(color_source)
      cv2.cuda.cvtColor(gpu_color_source, cv2.COLOR_BGR2LAB, gpu_color_source)
      source_img_stats = ct_gpu.image_stats_gpu(gpu_color_source)

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

  def callback(frame: np.ndarray, frame_id: int, color_calib_device='cpu') -> np.ndarray:
    global source_img_stats, out, fps_monitor, line_counts, args
    if args.color_calib_enable:
      ''' Color Calibration '''
      if args.color_calib_device == 'cpu':
        frame_cpu = ct_cpu.color_transfer_cpu(source_img_stats, frame, clip=False, preserve_paper=False)
        out.write(frame_cpu); #print(frame)
      elif args.color_calib_device == 'gpu':
        gpu_frame.upload(frame)
        frame_gpuMat = ct_gpu.color_transfer_gpu(source_img_stats, gpu_frame, clip=False, preserve_paper=False)
        frame_cpu = frame_gpuMat.download()
        out.write(frame_cpu)
        frame = ct_gpu.gpu_mat_to_torch_tensor(frame_gpuMat)
    else:
      frame_cpu = frame

    ''' Detection '''
    if args.yolo_version == 'yolov8':
      if MODEL_EXTENSION == "engine":
        bgr, ratio, dwdh = letterbox(frame_cpu, (W,H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # rgb = frame_cpu.copy()
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh*2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        data = Engine(tensor)
        # print(data)
        bboxes, scores, labels = det_postprocess(data)
        detections = sv.Detections.empty()
        detections.xyxy = ((bboxes-dwdh)/ratio).cpu().numpy()
        detections.confidence = scores.cpu().numpy()

        detections.class_id = labels.cpu().numpy().astype(int)
        # if detections.confidence.size > 0:
        #   print(detections)
      else:
        # yolov8
        results = yolo_model.predict(frame_cpu, verbose=False, iou=0.7, conf=0.1, device="cuda")[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, selected_classes)]

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
    for i, line_zone in enumerate(line_zones):
      objects_in, objects_out = line_zone.trigger(detections)
      for obj in detections.class_id[np.isin(objects_in, True)]:
        line_counts[i][CLASS_NAMES_DICT[obj]+"_in"] += 1
      for obj in detections.class_id[np.isin(objects_out, True)]:
        line_counts[i][CLASS_NAMES_DICT[obj]+"_out"] += 1

    fps_monitor.tick()
    ''' Log Time'''
    if frame_id % 20 == 0:
      logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, video_info.total_frames, max(1e-5, fps_monitor())))
      progress = frame_id / video_info.total_frames * 100  # Calculate progress as a percentage
      with open('progress.txt', 'w') as f:
        f.write(str(progress))
      logger.info(f"Write Progress: {progress}")

    # ''' Log Time'''
    # if frame_id % 20 == 0:
    #     progress = frame_id / video_info.total_frames * 100  # Calculate progress as a percentage
    #     yield progress  # Yield progress updates to the client
      
    return annotated_frame

  ''' Now process the whole video '''
  logger.info(f"saving results to {TRACK_OUTPUT_FILE_PATH}")

  sv.process_video( 
    source_path = SOURCE_VIDEO_PATH, 
    target_path = TARGET_VIDEO_PATH_ANN,
    callback=callback
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
  writer.writerow([f"Average FPS: {fps_monitor()}"])

