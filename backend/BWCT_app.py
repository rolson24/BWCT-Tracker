import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask import send_from_directory, send_file

from flask_uploads import UploadSet, configure_uploads, ALL
from flask import Response, stream_with_context
from flask_socketio import SocketIO

import atexit

from werkzeug.utils import secure_filename
import os
import subprocess
from os import remove
from os.path import exists
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import glob

import pandas as pd
import math

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cv2 as cv

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

class FileWatchHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        super(FileWatchHandler, self).on_modified(event)
        if not event.is_directory:
            self.callback(event.src_path)

    def on_moved(self, event):
        super(FileWatchHandler, self).on_moved(event)
        self.callback(event.dest_path)

def start_file_watching(path_to_watch, callback):
    event_handler = FileWatchHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    observer.start()
    return observer

def handle_file_change(file_path):
    print(f"File changed: {file_path}")
    # Implement your handling logic here


app = Flask(__name__)
socketio = SocketIO(app)

videos = UploadSet('videos', ALL)

video_path = None

app.config['UPLOADED_VIDEOS_DEST'] = 'static/uploads'
configure_uploads(app, videos)


@app.route('/', methods=['GET', 'POST'])
def upload():
    global video_path
    if request.method == 'POST':
        video = request.files.get('video')
        if video:
            filename = video.filename
            video_path = os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename)
            if not os.path.exists(video_path):
                video.save(video_path)
            # os.system(f"ffmpeg -i {video_path} -c:v libx264 -preset veryfast -crf 23 {app.config['UPLOADED_VIDEOS_DEST']}/{filename.split('.')[0]}.mp4")
            if exists('coordinates.txt'):
                remove('coordinates.txt')
            return jsonify({'message': 'Video uploaded', 'filename': filename})
        return jsonify({'message': 'No video provided'}), 400
    else:
        print("render template")
        return render_template('upload.html')  # replace 'index.html' with your actual template

@app.route('/receive-file-paths', methods=['POST'])
def receive_file_paths():
    global file_paths
    data = request.get_json()
    print(data)
    file_paths = data['filenames']
    print(file_paths)
    if exists('coordinates.txt'):
        remove('coordinates.txt')
    # Process the file paths as needed here
    for path in file_paths:
        observer = start_file_watching(path, handle_file_change)
    return jsonify({'message': 'Received file paths successfully'})

@app.route('/upload_day_night', methods=['GET', 'POST'])
def upload_day_night():
    global video_path
    if request.method == 'POST':
        day_night_file = request.files.get('day_night_file')
        if day_night_file:
            filename = secure_filename(day_night_file.filename)
            day_night_file_path = "static/day_night.csv"
            day_night_file.save(day_night_file_path)
            return jsonify({'message': 'File uploaded', 'filename': filename})
        return jsonify({'message': 'No file provided'}), 400
    else:
        print("render template")
        return render_template('upload.html')  # replace 'index.html' with your actual template
    

@app.route('/stream_video')
def stream_video():
    # filename = request.args.get('filename')  # Get filename from query parameter
    filename = file_paths[0]
    if not filename:
        return "Filename not provided", 400
    # filename = os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename)

    def generate():
        cmd = [
            'ffmpeg',
            '-i', filename,  # Use the dynamically provided filename
            '-f', 'mp4',
            '-vcodec', 'libx264',
            '-preset', 'veryfast',
            '-movflags', 'frag_keyframe+empty_moov',
            '-'
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            data = proc.stdout.read(4096)
            if not data:
                break
            yield data
        proc.wait()

    return Response(generate(), mimetype='video/mp4')

@app.route('/download_counts')
def download_counts():
    filename = os.path.split(file_paths[0])[0]
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('static/outputs', video_name)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # Sort the folders by creation time and get the most recent one
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None
    # Check if there is a most recent run folder
    if most_recent_run:
        # Construct the path to the counts file within the most recent run folder
        counts_file_path = os.path.join(most_recent_run, f'{video_name}_counts_output.txt')
        # Check if the counts file exists
        if os.path.exists(counts_file_path):
            return send_file(counts_file_path, as_attachment=True)
        else:
            return "Counts file not found in the most recent run", 404
    else:
        return "No runs found for the video", 404

@app.route('/download_processed_video')
def download_video():
    filename = os.path.split(file_paths[0])[0]
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    video_ext = filename.split('.')[1]
    base_path = os.path.join('static/outputs', video_name)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # Sort the folders by creation time and get the most recent one
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None
    # Check if there is a most recent run folder
    if most_recent_run:
        # Construct the path to the counts file within the most recent run folder
        counts_file_path = os.path.join(most_recent_run, f'{video_name}_annotated.{video_ext}')
        # Check if the counts file exists
        if os.path.exists(counts_file_path):
            return send_file(counts_file_path, as_attachment=True)
        else:
            return "Counts file not found in the most recent run", 404
    else:
        return "No runs found for the video", 404


@app.route('/download_lines')
def download_lines():
    return send_file('line_crossings.txt', as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOADED_VIDEOS_DEST'], filename)

@app.route('/coordinates', methods=['POST'])
def coordinates():
    line = request.form.get('line')
    if line:
        with open('coordinates.txt', 'a') as f:
            f.write(line + '\n')
        return jsonify({'message': 'Coordinates recorded'})
    return jsonify({'message': 'No coordinates provided'}), 400

@app.route('/clear_lines', methods=['POST'])
def clear_lines():
    if exists('coordinates.txt'):
        remove('coordinates.txt')
    return jsonify({'message': 'Lines cleared'})

def seconds_to_hms(seconds):
    if seconds == None:
        return "infinite"
    else:
        hours = int(seconds //  3600)
        minutes = int((seconds %  3600) //  60)
        seconds = int(seconds %  60)
        return "{:02}:{:02}:{:02}".format(hours, minutes, seconds)
def calculate_estimated_time_remaining(progress, processing_seconds):
    if progress != 0:
        estimated_total_time = (processing_seconds * 100) / progress
        estimated_time_remaining = estimated_total_time - processing_seconds

    else:
        estimated_time_remaining = None
    return seconds_to_hms(estimated_time_remaining)


@app.route('/process', methods=['POST'])
def process():
    # Use request.form or request.files to access the data
    data = request.json  # Get the JSON data sent with the POST request
    save_video = data.get('save_video', 'no')  # Use .get to provide a default value of 'no'
  
    if save_video == "yes":
        save_video = True
    elif save_video == "no":
        save_video = False
        
    filename = file_paths[0]


    print("Received filename:", filename)
    print("Save video option:", save_video)

    if filename:
        # Pass both filename and save_video to your processing function
        socketio.start_background_task(process_video, filename, save_video)
        return jsonify({'message': 'Video processing started'}), 200
    return jsonify({'message': 'No video file provided'}), 400

def process_video(filename, save_video):
    # Call your track.py script here
    print("Start processing video")
    tracker_base_path = "backend/Impr-Assoc-counter_demo"
    script_path = f"{tracker_base_path}/track.py"
    output_path = "backend/static/outputs/"
    model_path = f"{tracker_base_path}/models/yolov8s-2024-02-16-best_fp16_trt.engine"
    cc_source_path = f"{tracker_base_path}/reference-image-test.jpg"
    day_night_path = "static/day_night.csv"
    # model_path = "../Impr-Assoc-counter_demo/models/yolov8s-2024-02-14-best_fp16_trt.engine"

    # tracker = "Impr_Assoc"
    tracker = "ConfTrack"
    # video_path = os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename)
    video_path = filename
    try:
        if save_video:
            process = subprocess.Popen(
                [
                    "python", 
                    # "-m", "cProfile",
                    # "-s", "cumtime",
                    script_path,
                    "--source_video_path", video_path,
                    "--output_dir", output_path,
                    "-f", "coordinates.txt",
                    "-c", model_path,
                    "--save-frames",
                    # "--color_calib_enable",
                    "--color_source_path", cc_source_path,
                    "--color_calib_device", "cpu",
                    # "--day_night_switch_file", day_night_path,
                    "--object_tracker", tracker
                ],
                #  stdout=subprocess.PIPE,
                #  stderr=subprocess.PIPE
                )
        else:
            process = subprocess.Popen(
                [
                    "python", 
                    # "-m", "cProfile",
                    # "-s", "cumtime",
                    script_path,
                    "--source_video_path", video_path,
                    "--output_dir", output_path,
                    "-f", "coordinates.txt",
                    "-c", model_path,
                    # "--color_calib_enable",
                    "--color_source_path", cc_source_path,
                    "--color_calib_device", "cpu",
                    # "--day_night_switch_file", day_night_path,
                    "--object_tracker", tracker

                ],
                #  stdout=subprocess.PIPE,
                #  stderr=subprocess.PIPE
                )
        socketio.emit('progress', {'data': 0})
        processing_seconds = 0
        while process.poll() is None:
            try:
                with open('progress.txt', 'r') as f:
                    progress = float(f.read().strip())
                    socketio.emit('progress', {'data': progress, 'time': calculate_estimated_time_remaining(progress, processing_seconds)})
                    print(f"Progress: {progress}")
            except FileNotFoundError:
                print("File not found")
                pass  # File not found, continue waiting
            except ValueError:
                print("No valid  progress num")
                pass

            time.sleep(1)  # Wait for a short period before checking again
            processing_seconds += 1
        if os.path.exists('progress.txt'):
            os.remove('progress.txt')
        # stderr = process.stderr.read().decode('utf-8')
        # if stderr:
        #     print(f"Error: {stderr}")
        # After the subprocess has finished and the progress file has been deleted
        socketio.emit('video_processed', {'filename': filename})   
    except Exception as e:
        print(f"Error: {e}")
    
@app.route('/reprocess', methods=['POST'])
def reprocess():
    # data = request.get_json()
    # print("Received data:", data)  # Add this line to log the received data
    # filename = data.get('filename')
    filename = os.path.split(file_paths[0])[0]
    if filename:
        socketio.start_background_task(reprocess_video, filename)
        return jsonify({'message': 'Video re-processing started'}),  200
    return jsonify({'message': 'No video file provided'}),  400

def reprocess_video(filename):
    # Run the different script on the output_tracks.csv file\
    tracker_base_path = "backend/Impr-Assoc-counter_demo"
    reprocess_script_path = f"{tracker_base_path}/reprocess_tracks.py"
    base_path = "backend/static/outputs/"
    video_name = filename.split('.')[0]
    base_path = os.path.join(base_path, video_name)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # Sort the folders by creation time and get the most recent one
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None


    # Check if there is a most recent run folder
    if most_recent_run:
        tracks_input_file = os.path.join(most_recent_run, f'{video_name}_tracks_output.txt')
        output_counts_path = os.path.join(most_recent_run, f'{video_name}_counts_output.txt') # overwrite the counts file
        line_crossings_path = os.path.join(most_recent_run, f'{video_name}_line_crossings.txt')
        # Check if the tracks file exists
        print(tracks_input_file)
        if os.path.exists(tracks_input_file):
            try:
                    # Call the reprocess script
                process = subprocess.Popen(['python3', reprocess_script_path,
                                    '--count_lines_file', 'coordinates.txt', 
                                    '--tracks_input_file', tracks_input_file,
                                    '--output_counts_file', output_counts_path,
                                    '--line_crossings', line_crossings_path
                                    ],
                                    # stdout=subprocess.PIPE,
                                    # stderr=subprocess.PIPE
                                    )
                socketio.emit('progress', {'data': 0})

                processing_seconds = 0

                while process.poll() is None:
                    # print("Waiting for progress")
                    try:
                        with open('progress.txt', 'r') as f:
                            progress = float(f.read().strip())
                            socketio.emit('progress', {'data': progress, 'time': calculate_estimated_time_remaining(progress, processing_seconds)})
                            print(f"Progress: {progress}")
                    except FileNotFoundError:
                        print("Progress file not found")
                        pass  # File not found, continue waiting
                    except ValueError:
                        print("No valid progress num")
                        pass
                    # Read from stdout and stderr
                    # stdout = process.stdout.readline().decode('utf-8')
                    # stderr = process.stderr.readline().decode('utf-8')

                    # # Print to server logs
                    # if stdout:
                    #     print(f"STDOUT: {stdout}")
                    # if stderr:
                    #     print(f"STDERR: {stderr}")

                    time.sleep(1)  # Wait for a short period before checking again
                    processing_seconds += 1
                if os.path.exists('progress.txt'):
                    # Progress file exists, emit 100% progress and delete the file
                    socketio.emit('progress', {'data': 100})
                    os.remove('progress.txt')
                # stderr = process.stderr.read().decode('utf-8')
                # if stderr:
                #     print(f"Error: {stderr}")
                # After the subprocess has finished and the progress file has been deleted
                socketio.emit('video_processed', {'filename': filename})   
            except Exception as e:
                print(f"Error: {e}")
        else:  
            print("Tracks file not found in the most recent run")



@app.route('/counts', methods=['GET'])
def get_counts():
    filename = os.path.split(file_paths[0])[1]
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('backend/static/outputs', video_name)
    print(base_path)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # Sort the folders by creation time and get the most recent one
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None
    # Check if there is a most recent run folder
    if most_recent_run:
        # Construct the path to the counts file within the most recent run folder
        counts_file_path = os.path.join(most_recent_run, f'{video_name}_counts_output.txt')
        
        # Check if the counts file exists
        if os.path.exists(counts_file_path):
            # Read the counts file
            with open(counts_file_path, 'r') as f:
                counts_data = f.read()
            
            # Return the counts data as a JSON response
            return jsonify({'counts': str(counts_data), 'filename': str(video_name)})
        else:
            return jsonify({'message': 'Counts file not found in the most recent run'}),  404
    else:
        return jsonify({'message': 'No runs found for the video'}),  404

@app.route('/get_crossings_data', methods=['POST'])
def get_crossings_data():
    filename = os.path.split(file_paths[0])[1]
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('backend/static/outputs', video_name)
    print(base_path)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # Sort the folders by creation time and get the most recent one
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None
    # Check if there is a most recent run folder
    if most_recent_run:
        data = request.json
        fps = data['fps']  # Get fps from the request data

        # Read the line_crossings.txt file and parse the data
        line_crossings_file_path = os.path.join(most_recent_run, f'{video_name}_line_crossings.txt')
        crossings_df = pd.read_csv(line_crossings_file_path, header=None, skiprows=1, sep=',')
        crossings_df.columns = ['frame_num', 'line_num', 'class_name', 'direction']
        
        # Calculate timestamps and aggregate by hour
        crossings_df['timestamp'] = pd.to_datetime(crossings_df['frame_num'] / fps, unit='s')

        # Format timestamps to only include the time portion
        crossings_df['time'] = crossings_df['timestamp'].dt.time

        crossings_df.set_index('timestamp', inplace=True)
        hourly_counts = crossings_df.groupby([pd.Grouper(freq='15min'), 'line_num', 'class_name', 'direction']).size().reset_index(name='count')

        # Use the formatted 'time' for plotting
        hourly_counts['time'] = hourly_counts['timestamp'].dt.time
        hourly_counts.drop(columns=['timestamp'], inplace=True)

        # Convert to JSON or a suitable format for the frontend
        plotly_data = hourly_counts.to_json(orient='records', date_format='iso')
        
        return jsonify(plotly_data)

    else:
        return jsonify({'message': 'No runs found for the video'}),  404


def get_tracks(file_name):
    """"Use it to get the tracks into a dictionary, then calc the counts over the lines."""
    track_df = pd.read_csv(file_name, header=None)

    frame_index = 1
    # train_labels_1 is a pandas dataframe with columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, -1, -1
    track_df = track_df.sort_values([1, 0]) # sort by id first then frame second
    print(track_df.head())

    # Initialize an empty dictionary
    bounding_boxes = {}

    # Group the DataFrame by 'track_id' and iterate over each group
    for track_id, group in track_df.groupby(1):
        # Create a list of tuples for the bounding box coordinates
        # put into numpy array
        # print(group)
        boxes = np.array(list(zip(group[2], group[3], group[4], group[5], group[7])))

        # Assign this list to the corresponding track_id in the dictionary
        bounding_boxes[track_id] = boxes

    # Display the resulting dictionary
    # print(bounding_boxes)
    return bounding_boxes

def save_middle_video_frame(file_path, save_path):
    """Takes a video, extracts the middle frame and saves it into save_path
    Also returns the video resolution"""
    # Open the video file
    cap = cv.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    else:
        # Get the width and height of the video frames
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        resolution = (frame_width, frame_height)

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        # Calculate the frame number for the middle frame
        middle_frame_number = total_frames // 2
        
        # Set the current video position to the middle frame
        cap.set(cv.CAP_PROP_POS_FRAMES, middle_frame_number)
        
        # Read the middle frame
        ret, frame = cap.read()
        
        if ret:
            # Save or display the frame
            cv.imwrite(f'{save_path}', frame)  # Save the frame as an image

        else:
            print("Error: Could not read the middle frame.")
        
        # Release the video capture object
        cap.release()
    return resolution

@app.route('/get_tracks', methods=['POST'])
def get_track_data_plot():
    filename = os.path.split(file_paths[0])[1]
    video_path = file_paths[0]
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('backend/static/outputs', video_name)
    print(base_path)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # Sort the folders by creation time and get the most recent one
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None
    # Check if there is a most recent run folder
    if most_recent_run:
        run_name = os.path.basename(most_recent_run)
        # Read the tracks.txt file and parse the data
        tracks_file_path = os.path.join(most_recent_run, f'{video_name}_tracks_output.txt')
        if os.path.exists(tracks_file_path):
            # tracks is a dict from track_ids to their array of bboxes in order of frame
            tracks = get_tracks(tracks_file_path)
            # Get the source video path
            # stored in static/uploads/
            # video_path = os.path.join('static/uploads', filename)
            saved_frame = f"{most_recent_run}/middle_frame.jpg"
            resolution = save_middle_video_frame(video_path, saved_frame)
            print(f"saved_frame {saved_frame}")
            saved_frame = f"/static/outputs/{video_name}/{run_name}/middle_frame.jpg"
            # saved_frame = os.path.join(os.path.curdir, saved_frame)
            print(f"saved_frame frontend {saved_frame}")
            kernel_size = 10
            kernel = np.ones(kernel_size) / kernel_size

            # Create subplots
            fig = make_subplots(rows=1, cols=1, subplot_titles=("10 frame smoothing"))

            for track_id in tracks.keys():
                trace = tracks[track_id]
                if len(trace) != 0:
                    # annotate the center of the bottom line of the BBox, because that is intuitive
                    trace_center_x = trace[:,0] + trace[:,2] // 2
                    trace_bottom_y = trace[:,1] + trace[:,3]
                    trace_convolved_x = np.convolve(trace_center_x, kernel, mode='valid')
                    trace_convolved_y = np.convolve(trace_bottom_y, kernel, mode='valid')
                    # Flip the y coords because images are index top to bottom
                    trace_convolved_y = trace_convolved_y

                    # # Try to merge lines into clusters
                    # # max difference between lines to be considered neighbors
                    # my_epsilon = 100

                    # # minimum number of lines to create a new cluster
                    # my_min_samples = 3
                    # # Get the end points of each track
                    # end_points = np.array([
                    #     (trace[-1, 0] + trace[-1, 2] // 2, trace[-1, 1] + trace[-1, 3]) 
                    #     for trace in tracks.values() if len(trace) > 1
                    # ])

                    # # Run DBSCAN clustering on the collected end points
                    # db = DBSCAN(eps=my_epsilon, min_samples=my_min_samples).fit(end_points)
                    # labels = db.labels_

                    # # Create a dictionary to hold lines for each cluster
                    # clustered_lines = {label: [] for label in set(labels) if label != -1}

                    # # Add lines to their respective clusters based on labels
                    # for label, trace in zip(labels, tracks.values()):
                    #     if label != -1:
                    #         clustered_lines[label].append(trace)

                    # # Draw a thicker line for each cluster
                    # for cluster_id, lines in clustered_lines.items():
                    #     if not lines:  # Skip if there are no lines in the cluster
                    #         continue

                    #     # Calculate the mean start and end points for the cluster
                    #     mean_start = np.mean([[line[0, 0] + line[0, 2] // 2, line[0, 1] + line[0, 3]] for line in lines], axis=0)
                    #     mean_end = np.mean([[line[-1, 0] + line[-1, 2] // 2, line[-1, 1] + line[-1, 3]] for line in lines], axis=0)
                        
                    #     # Add trace for the cluster
                    #     fig.add_trace(go.Scatter(
                    #         x=[mean_start[0], mean_end[0]], 
                    #         y=[mean_start[1], mean_end[1]], 
                    #         mode='lines',
                    #         line=dict(width=3),  # Adjust line thickness
                    #         name=f'Cluster {cluster_id}'
                    #     ))

                    # Add smoothed path
                    fig.add_trace(go.Scatter(x=trace_convolved_x, y=trace_convolved_y, mode='lines', name=f'Trace {track_id} smoothed'), row=1, col=1)
                    # Add an arrow annotation to the end of each trace
                    # if len(trace_convolved_x) > 10:
                    #     fig.add_annotation(
                    #         x=trace_convolved_x[-1],  # X-coordinate of the arrow's head (end of the line)
                    #         y=trace_convolved_y[-1],  # Y-coordinate of the arrow's head (end of the line)
                    #         ax=trace_convolved_x[-10],  # X-coordinate of the arrow's tail
                    #         ay=trace_convolved_y[-10],  # Y-coordinate of the arrow's tail
                    #         xref='x', yref='y',
                    #         axref='x', ayref='y',
                    #         text='',  # No text
                    #         showarrow=True,
                    #         arrowhead=3,
                    #         arrowsize=1,
                    #         arrowwidth=2,
                    #         arrowcolor='red'
                    #     )
            # Update xaxis and yaxis properties for each subplot
            fig.update_xaxes(title_text="X", range=[0, resolution[0]], row=1, col=1)
            fig.update_yaxes(title_text="Y", range=[resolution[1], 0], row=1, col=1)

            # aspect ratio of image
            # aspect_ratio = resolution[0]/resolution[1]
            # plot_width = 800
            # plot_height = int(plot_width / aspect_ratio)
            # Set a background image
            fig.update_layout(
                images=[
                    dict(
                        source=saved_frame,
                        xref="x", yref="y",  # Use "paper" to refer to the whole plotting area
                        x=0, y=1,  # These specify the position of the image (0,0 is bottom left, 1,1 is top right)
                        sizex=resolution[0], sizey=resolution[1],  # These specify the size of the image. 1,1 will cover the entire background
                        sizing="stretch",
                        opacity=1.0,  # Set the opacity of the image
                        layer="below"  # Ensure the image is below the plot
                    )
                ]
            )

            # Update layout if needed
            fig.update_layout(height=resolution[1], width=resolution[0], title_text="Path Analysis", title_x=0.5)

            fig_json = fig.to_json()
            return jsonify(fig_json=fig_json)
        else:
            return jsonify({'message': 'No tracks found for this video'}), 404
    else:
        return jsonify({'message': 'No runs found for the video'}),  404

def shutdown_handler():
    """Handle shutdown. Delete coordinates file"""
    if exists('coordinates.txt'):
        remove('coordinates.txt')

if __name__ == '__main__':
    print("start webapp")
    atexit.register(shutdown_handler)
    app.run(debug=False, port=5000)