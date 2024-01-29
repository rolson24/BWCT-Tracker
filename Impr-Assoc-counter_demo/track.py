import argparse
import os
import sys
import os.path as osp
import cv2
import numpy as np
import torch

sys.path.append('.')

from loguru import logger

import supervision as sv

# from yolox.data.data_augment import preproc
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess
# from yolox.utils.visualize import plot_tracking

# from tracker.tracking_utils.timer import Timer
# from tracker.Impr_Assoc_Track import ImprAssocTrack

from ultralytics import YOLO

import color_transfer_cpu as ct

import csv

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
	parser = argparse.ArgumentParser("Track and Count People with Improved Association Track!")

	parser.add_argument("--source_video_path", help="path to source video to perform counting on. 'path/to/video.ext' ext can be: ('mp4', 'm4v', 'mjpeg', 'avi', 'h264')")
	parser.add_argument("--color_source_path", help="path to color source image for color correction. 'path/to/image.ext' ext must be: ('jpg')")
	parser.add_argument("--output_dir", help="path to target output directory. 'path/to/output/dir'")

#     parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
#     parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
	parser.add_argument("-f", "--count_lines_file", default=None, type=str, help="input your count lines filepath (format specified in docs)")
	parser.add_argument("-c", "--ckpt", default=None, type=str, help="path to yolo weights file")
	parser.add_argument("-expn", "--experiment-name", type=str, default=None)
	parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
	# parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

	# Detector
	parser.add_argument("--yolo_version", default="yolov8", type=str, help="yolo model architecture. Can be 'yolov8' or 'yolo-nas'")
	parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
	parser.add_argument("--conf", default=None, type=float, help="test conf")
	parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
	parser.add_argument("--tsize", default=None, type=int, help="test img size")
	parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
	parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

	# tracking args
	parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
	parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
	parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
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
	parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
	parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"./fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
	parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"./models/mot17_sbs_S50.pth", type=str, help="reid config file path")
	parser.add_argument('--proximity_thresh', type=float, default=0.1, help='threshold for rejecting low overlap reid matches')
	parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

	return parser


def parse_count_lines_file(count_lines_file):
	'''File format:
		(x1, y1) (x2, y2)
		(x1, y1) (x2, y2)'''
	with open(count_lines_file, 'r') as f:
		lines = f.readlines()
	lines = [line.strip() for line in lines]
	lines = [line.split(' ') for line in lines]
	lines = [[eval(coord.replace('(', '').replace(')', '')) for coord in line] for line in lines]
	return lines
	

if __name__ == "__main__":
	args = make_parser().parse_args()

	SOURCE_VIDEO_PATH = args.source_video_path
	SOURCE_VIDEO_NAME = osp.basename(SOURCE_VIDEO_PATH).split('.')[0]
	OUTPUT_DIR = args.output_dir
	TARGET_VIDEO_PATH_ANN = f"{OUTPUT_DIR}/{SOURCE_VIDEO_NAME}_annotated.mp4"
	TARGET_VIDEO_PATH_CLEAN = f"{OUTPUT_DIR}/{SOURCE_VIDEO_NAME}_clean.mp4"; print(TARGET_VIDEO_PATH_CLEAN)
	TRACK_OUTPUT_FILE_PATH = f"{OUTPUT_DIR}/{SOURCE_VIDEO_NAME}_track_output.txt"
	COUNT_OUTPUT_FILE_PATH = f"{OUTPUT_DIR}/{SOURCE_VIDEO_NAME}_count_output.txt"
	COLOR_SOURCE_PATH = args.color_source_path
	COUNT_LINES_FILE = args.count_lines_file


	if args.yolo_version == 'yolov8':
		''' load YOLOv8'''
		# change this
		MODEL = args.ckpt

		# load YOLOv8
		logger.info("loading YOLOv8 model from: {}", MODEL)
		yolo_model = YOLO(MODEL)
		yolo_model.fuse()
	elif args.yolo_version == 'yolo-nas':
		''' load YOLO-NAS'''
		# change this
		MODEL = args.ckpt

		# load YOLO-NAS
		logger.info("loading YOLO-NAS model from: {}", MODEL)

		yolo_model = YOLO(MODEL)
		yolo_model.fuse()

	# dict maping class_id to class_name
	CLASS_NAMES_DICT = yolo_model.model.names

	# class_ids of interest - pedestrians, bikes, scooters, wheelchairs
	selected_classes = [0,1,2,3]
	print(CLASS_NAMES_DICT)

	video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

	count_lines = parse_count_lines_file(COUNT_LINES_FILE)
	line_zones = []
	# for line in count_lines:
	line_zones.append(sv.LineZone(start=sv.Point(640, 0), end=sv.Point(640, 720)))
	print(f"line_zones {line_zones}")

	# impr_assoc_tracker = ImprAssocTrack(track_high_thresh=args.track_high_thresh,
	#                                     track_low_thresh=args.track_low_thresh,
	#                                     new_track_thresh=args.new_track_thresh,
	#                                     track_buffer=args.track_buffer,
	#                                     match_thresh=args.match_thresh,
	#                                     second_match_thresh=args.second_match_thresh,
	#                                     aspect_ratio_thresh=args.aspect_ratio_thresh,
	#                                     min_box_area=args.min_box_area,
	#                                     overlap_thresh=args.overlap_thresh,
	#                                     iou_weight=args.iou_weight,
	#                                     proximity_thresh=args.proximity_thresh,
	#                                     appearance_thresh=args.appearance_thresh,
	#                                     with_reid=args.with_reid,
	#                                     fast_reid_config=args.fast_reid_config,
	#                                     fast_reid_weights=args.fast_reid_weights,
	#                                     device=args.device,
	#                                     frame_rate=video_info.fps)

	impr_assoc_tracker = sv.ByteTrack(track_thresh=0.25,
									 track_buffer=20,
									 match_thresh=0.8,
									 frame_rate=video_info.fps
									 )

	out = cv2.VideoWriter(TARGET_VIDEO_PATH_CLEAN, cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps, (video_info.width, video_info.height))

	# create instance of BoxAnnotator
	box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=2, text_scale=1)

	# create instance of TraceAnnotator
	trace_annotator = sv.TraceAnnotator(thickness=1, color=(0, 255, 0), trace_length=300)

	# create instance of LineZoneAnnotator
	line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)
	
	# create instance of FPSMonitor
	fps_monitor = sv.FPSMonitor()

	'''for cpu color correction'''
	color_source = cv2.imread(COLOR_SOURCE_PATH)
	color_source = cv2.cvtColor(color_source, cv2.COLOR_BGR2RGB).astype(np.float32)
	source_img_stats = ct.image_stats(color_source)


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
	
	def callback(frame: np.ndarray, frame_id: int, color_calib_device='cpu') -> np.ndarray:
		global source_img_stats, out, fps_monitor, line_counts
		fps_monitor.tick()

		''' Color Calibration '''
		if color_calib_device == 'cpu':
			frame = ct.color_transfer_cpu(source_img_stats, frame, clip=False, preserve_paper=False)
			out.write(frame); print(frame)
		# elif color_calib_device == 'gpu':
		#     gpu_frame.upload(frame)
		#     cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2LAB, gpu_frame)
		#     frame = image_transfer_gpu(source_img_stats, gpu_frame, clip=False, preserve_paper=False)
		#     frame = frame.download()
		#     out.write(frame)

		''' Detection '''
		if args.yolo_version == 'yolov8':
			# yolov8
			results = yolo_model(frame, verbose=False, iou=0.7, conf=0.1)[0]
			detections = sv.Detections.from_ultralytics(results)
			detections = detections[np.isin(detections.class_id, selected_classes)]
		elif args.yolo_version == 'yolo-nas':
			results = yolo_model.predict(frame, iou=0.7, conf=0.1, fuse_model=False)[0]
			detections = sv.Detections.from_yolo_nas(results)
			detections = detections[np.isin(detections.class_id, selected_classes)]

		''' Tracking '''
		# commented one for impr_associated
		# detections = impr_assoc_tracker.update_with_detections(detections, frame)
		detections = impr_assoc_tracker.update_with_detections(detections)

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
			annotated_frame = frame.copy()
		else:
			annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
		
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
		out.release()
		return annotated_frame

	''' Now process the whole video '''
	logger.info(f"saving results to {TRACK_OUTPUT_FILE_PATH}")

	sv.process_video(
		source_path = SOURCE_VIDEO_PATH,
		target_path = TARGET_VIDEO_PATH_ANN,
		callback=callback
	)

	''' Save Tracks '''
	with open(COUNT_OUTPUT_FILE_PATH, 'a+', newline='', encoding='UTF8') as f:
		writer = csv.writer(f)
		for i, line_count in enumerate(line_counts):
			writer.writerow([f"line {i}"])
			for key, val in line_count.items():
				writer.writerow([key, val])

