from loguru import logger
import os
import supervision as sv
import argparse

from supervision.detection.line_counter import LineZone
from supervision.geometry.core import Point, Position, Vector
from supervision.detection.core import Detections

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

import pandas as pd
import csv

def make_parser():
    parser = argparse.ArgumentParser("Reprocess tracks to count line crossings.")
    parser.add_argument('--tracks_input_file', type=str, help="File containing tracks to reprocess.", required=True)
    parser.add_argument('--count_lines_file', type=str, help="File containing count lines.", required=True)
    parser.add_argument('--output_counts_file', type=str, help="Path to output file containing the total counts.", required=True)
    parser.add_argument('--line_crossings', type=str, help="Path to output file containing the time-stamped counts.", required=True)
    return parser


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
    line_ends = []
    logger.info(f"lines post replace: {lines}")
    for i in range(len(lines)):
        line_ends.append((Point(*lines[i][0]), Point(*lines[i][1])))
    return line_ends

def get_frames(file_name):

  track_df = pd.read_csv(file_name, header=None)

  frame_index = 1
  # train_labels_1 is a pandas dataframe with columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, -1, -1
  track_df = track_df.sort_values([0, 1]) # sort by frame then id
  print(track_df.head())

  # Initialize an empty dictionary
  bounding_boxes = {}

  # Group the DataFrame by 'frame' and iterate over each group
  for frame_id, group in track_df.groupby(0):
      # Create a list of tuples for the bounding box coordinates
      # put into numpy array
      # print(group)
      boxes = np.array(list(zip(group[2], group[3], group[4], group[5], group[7], group[1])))

      # Assign this list to the corresponding frame_id in the dictionary
      bounding_boxes[frame_id] = boxes

  # Display the resulting dictionary
  # print(bounding_boxes)
  return bounding_boxes

def count_lines_supervision(line_ends, bboxes, LINE_CROSSINGS_FILE):

  line_zones = []
  with open(LINE_CROSSINGS_FILE, 'w') as f:
    # f.write(f"\n")
    # Change to use BOTTOM_CENTER as the trigger for a count (more intuitive)
    triggering_anchors = [sv.Position.BOTTOM_CENTER]
    for i, end_points in enumerate(line_ends):
        line_zones.append(LineZone(start=end_points[0], end=end_points[1], triggering_anchors=triggering_anchors))
        # write the line coordinates to the file with a comma after each one.
        # format: line_0: ((x1, y1>, <x2, y2>), line_1: (<x1, y1>, <x2, y2>), line_2: (<x1, y1>, <x2, y2>), line_3: (<x1, y1>, <x2, y2>) ...
        f.write(f"line_{i}: ({end_points[0].as_xy_int_tuple()}, {end_points[1].as_xy_int_tuple()}),")
    f.write("\n")

  line_counts = []
  for i in range(len(line_zones)):
    class_counts = {}
    for val in CLASS_NAMES_DICT.values():
      class_counts[val+"_in"] = 0
      class_counts[val+"_out"] = 0
    line_counts.append(class_counts)

  ''' Bboxes should be a dict of type {frame_id:(tlx, tly, width, height, class_id, track_id)} '''
  # A, B, C = line_equation(x1, y1, x2, y2)

  # counts = {"0_out":0, "0_in":0, "1_out":0, "1_in":0, "2_out":0, "2_in":0, "3_out":0, "3_in":0, }
  for key, value in bboxes.items():
    if key < 15:  # Replace 10 with the actual value you want to use
        logger.info(f"{key}: {value}")
  num_frames = len(bboxes)
  for j, frame_ind in enumerate(bboxes.keys()):
    '''xyxy (np.ndarray): An array of shape `(n, 4)` containing
    the bounding boxes coordinates in format `[x1, y1, x2, y2]` '''
    # logger.info(f"{bboxes}")
    frame = bboxes[frame_ind]
    # logger.info("{} {}".format(frame_ind, frame))
    x1 = frame[:,0]
    y1 = frame[:,1]
    width = frame[:,2]
    height = frame[:,3]
    frame[:,2] = x1 + width
    frame[:,3] = y1 + height
    class_ids = frame[:,4]
    dets = Detections(xyxy=frame[:,0:4], class_id=class_ids, tracker_id=frame[:,5])
    # print(dets)


    # Need to write to a file that stores the counts in this format:
    # <frame_id>,<line number>,<class name>,<in/out>
    for i, line_zone in enumerate(line_zones):
        objects_in, objects_out = line_zone.trigger(dets)
        for obj in dets.class_id[np.isin(objects_in, True)]:
            line_counts[i][CLASS_NAMES_DICT[obj]+"_in"] += 1
            with open(LINE_CROSSINGS_FILE, 'a') as f:
                f.write(f"{frame_ind},{i},{CLASS_NAMES_DICT[obj]},in\n")
        for obj in dets.class_id[np.isin(objects_out, True)]:
            line_counts[i][CLASS_NAMES_DICT[obj]+"_out"] += 1
            with open(LINE_CROSSINGS_FILE, 'a') as f:
                f.write(f"{frame_ind},{i},{CLASS_NAMES_DICT[obj]},out\n")
    
    if j % 20 == 0:
      # logger.info('Processing track frame {}/{}'.format(j, num_frames))
      progress = j / num_frames * 100  # Calculate progress as a percentage
      with open('progress.txt', 'w') as f:
        f.write(str(progress))
    #   logger.info(f"Write Progress: {progress}")

  return line_counts

args = make_parser().parse_args()
# logger.info(f"Args: {args}")
COUNT_LINES_FILE = args.count_lines_file
CLASS_NAMES_DICT = {1:"Pedestrians", 0:"Bikes", 2:"Scooters", 3:"Wheelchairs"} # for yolov8n-2023-11-03 change for yolov8s-2024-02-14
DATA_INPUT = args.tracks_input_file
DATA_OUTPUT = args.output_counts_file
LINE_CROSSINGS_FILE = args.line_crossings

''' Get Frames '''
# logger.info(f"Getting frames")
frames = get_frames(DATA_INPUT)
# logger.info(f"Got frames")
''' Get Line Ends '''
line_ends = parse_count_lines_file(COUNT_LINES_FILE)
# logger.info(f"Got line ends: {line_ends}")
''' Count Lines '''
line_counts = count_lines_supervision(line_ends, frames, LINE_CROSSINGS_FILE)

''' Save Counts ''' 
with open(DATA_OUTPUT, 'w', newline='', encoding='UTF8') as f:
  writer = csv.writer(f)
  for i, line_count in enumerate(line_counts):
    writer.writerow([f"line {i}"])
    for key, val in line_count.items():
      writer.writerow([key, val])

