import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask import send_from_directory, send_file

from flask_uploads import UploadSet, configure_uploads, ALL
from flask import Response, stream_with_context
from flask_socketio import SocketIO
import zipfile
import logging

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
import json

import pandas as pd
import math

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import re

import cv2 as cv
from PIL import Image

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
    app.logger.debug(f"File changed: {file_path}")
    # Implement your handling logic here



app = Flask(__name__)
socketio = SocketIO(app)

videos = UploadSet('videos', ALL)

video_path = None

app.config['UPLOADED_VIDEOS_DEST'] = 'static/uploads'
configure_uploads(app, videos)



track_fig = None
volume_fig = None
crossings_fig = None
counts_fig = None

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
            if exists('backend/coordinates.txt'):
                remove('backend/coordinates.txt')
            return jsonify({'message': 'Video uploaded', 'filename': filename})
        return jsonify({'message': 'No video provided'}), 400
    else:
        app.logger.debug("render template")
        return render_template('frontend.html')  # replace 'index.html' with your actual template
    
@app.route('/health')
def health_check():
    return "OK", 200

@app.route('/receive-file-paths', methods=['POST'])
def receive_file_paths():
    global file_paths
    data = request.get_json()
    app.logger.debug(data)
    file_paths = data['filenames']
    app.logger.debug(file_paths)
    if exists('backend/coordinates.txt'):
        remove('backend/coordinates.txt')
    # Process the file paths as needed here
    # for path in file_paths:
    #     observer = start_file_watching(path, handle_file_change)
    app.logger.debug("got file paths")
    return jsonify({'message': f'Received file paths successfully: {file_paths}'})

@app.route('/receive-raw-tracks-file-path', methods=['POST'])
def receive_raw_tracks_file_path():
    global file_paths
    data = request.get_json()
    app.logger.debug(data)
    file_paths = data['filenames']
    app.logger.debug(file_paths)
    if exists('backend/coordinates.txt'):
        remove('backend/coordinates.txt')
    # Process the file paths as needed here
    
    zip_file = file_paths[0]

    video_name = os.path.basename(zip_file).split('.')[0]
    app.logger.debug(f"zip file name: {video_name}")
    # base_path = os.path.join(base_path, video_name)
    # run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # # Sort the folders by creation time and get the most recent one
    # run_folders.sort(key=os.path.getctime, reverse=True)
    # most_recent_run = run_folders[0] if run_folders else None

    # Specify the directory to extract to
    extract_to = f'backend/static/outputs/{video_name}/run_1'

    tracks_file = os.path.join(extract_to, "tracks_output.txt")
    tracks_file_new = os.path.join(extract_to, f"{video_name}_tracks_output.txt")

    volume_file = os.path.join(extract_to, "person_volume.txt")
    volume_file_new = os.path.join(extract_to, f"{video_name}_person_volume.txt")


    # Ensure the target directory exists
    os.makedirs(extract_to, exist_ok=True)

    # Open the ZIP file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract all the contents into the directory
        zip_ref.extractall(extract_to)

    # Rename file to be in expected format for /get_tracks
    if os.path.exists(tracks_file):
        os.rename(tracks_file, tracks_file_new)

    # Rename file to be in expected format for /volume_file
    if os.path.exists(volume_file):
        os.rename(volume_file, volume_file_new)
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
        app.logger.debug("render template")
        return render_template('frontend.html')  # replace 'index.html' with your actual template
    

@app.route('/stream_video')
def stream_video():
    # filename = request.args.get('filename')  # Get filename from query parameter
    filename = file_paths[0]
    if not filename:
        return "Filename not provided", 400
    # filename = os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename)

    app.logger.debug(f"filename to reencode: {filename}")
    def generate():
        cmd = [
            'ffmpeg',
            '-i', filename,  # Use the dynamically provided filename
            '-loop', '1',
            '-f', 'mp4',
            '-vcodec', 'libx264',
            '-preset', 'veryfast',
            '-movflags', '+frag_keyframe+empty_moov+faststart',
            '-',
        ]
        # ffmpeg -i input.avi -c:v libx264 -c:a aac -movflags +faststart output.mp4

        app.logger.debug(f"cmd: {cmd}")
        try:
            if os.name == 'nt':
                app.logger.debug("On windows")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
            else:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            app.logger.debug(f"Started FFMPEG")
            # i = 0
            # stdout, stderr = proc.communicate()
            # app.logger.debug(f"FFmpeg error: {stderr.decode()}")
            # app.logger.debug(f"FFmpeg output: {stdout}")

            while True:
                data = proc.stdout.read(512)
                # proc.stderr.flush()
            
                if not data:
                    app.logger.debug("Done with video.")
                    break
                yield data
                # err = proc.stderr.read(1024).decode()
                # app.logger.debug(f"{err}")
                # i += 1
                
            stdout, stderr = proc.communicate()
            app.logger.debug(f"FFmpeg error: {stderr.decode()}")

            if proc.returncode != 0:
                app.logger.debug(f"FFmpeg error: {stderr.decode()}")
            proc.wait()

        except Exception as e:
            app.logger.debug("Error executing FFmpeg:", str(e))

    return Response(generate(), mimetype='video/mp4')

    
@app.route('/download_counts')
def download_counts():
    if not file_paths:  # Check if file_paths is empty
        return "No video file provided", 400
    
    filename = os.path.basename(file_paths[0])  # Get the filename
    video_name = filename.split('.')[0]  # Extract video name without extension
    base_path = os.path.join('backend/static/outputs', video_name)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None

    if most_recent_run:
        counts_file_path = os.path.join(most_recent_run, f'{video_name}_counts_output.txt')
        if os.path.exists(counts_file_path):
            return send_file(counts_file_path, as_attachment=True)
        else:
            return "Counts file not found in the most recent run", 404
    else:
        return "No runs found for the video", 404

@app.route('/get_raw_tracks_file_path', methods=['GET'])
def get_raw_tracks():
    filename = os.path.split(file_paths[0])[1]
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_name = 'backend/static/outputs'
    base_path = os.path.join(base_name, video_name)
    app.logger.debug(base_path)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # Sort the folders by creation time and get the most recent one
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None
    # Check if there is a most recent run folder
    if most_recent_run:
        counts_file_path = os.path.join(most_recent_run, f'{video_name}_tracks_output.txt')
        middle_video_frame_path = os.path.join(most_recent_run, "middle_frame.jpg")
        person_volume_file_path = os.path.join(most_recent_run, f'{video_name}_person_volume.txt')


        # Specify the name of the output ZIP file
        output_zip = os.path.join(base_name, f"{video_name}_raw_data.zip")

        # Create a new ZIP file
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as myzip:
            myzip.write(counts_file_path, arcname='tracks_output.txt')
            myzip.write(middle_video_frame_path, arcname='middle_frame.jpg')
            myzip.write(person_volume_file_path, arcname='person_volume.txt')


        if os.path.exists(output_zip):
            return output_zip
        else:
            return "Tracks file not found in the most recent run", 404
    else:
        return "No runs found for the video", 404


@app.route('/get_counts_file_path')
def get_counts_file_path():
    if not file_paths:  # Check if file_paths is empty
        return "No video file provided", 400
    
    filename = os.path.basename(file_paths[0])  # Get the filename
    video_name = filename.split('.')[0]  # Extract video name without extension
    base_path = os.path.join('backend/static/outputs', video_name)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None

    if most_recent_run:
        counts_file_path = os.path.join(most_recent_run, f'{video_name}_counts_output.txt')
        if os.path.exists(counts_file_path):
            return counts_file_path
        else:
            return "Counts file not found in the most recent run", 404
    else:
        return "No runs found for the video", 404


@app.route('/download_processed_video')
def download_video():
    filename = os.path.basename(file_paths[0])  # Get the filename
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    video_ext = filename.split('.')[1]
    base_path = os.path.join('backend/static/outputs', video_name)
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

@app.route('/get_processed_video_file_path')
def get_processed_video_file_path():
    filename = os.path.basename(file_paths[0])  # Get the filename
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    video_ext = filename.split('.')[1]
    base_path = os.path.join('backend/static/outputs', video_name)
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
            return counts_file_path
        else:
            return "Counts file not found in the most recent run", 404
    else:
        return "No runs found for the video", 404

@app.route('/download_lines')
def download_lines():
    return send_file('line_crossings.txt', as_attachment=True)

@app.route('/get_line_crossings_file_path')
def get_line_crossings_file_path():
    filename = os.path.basename(file_paths[0])  # Get the filename
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('backend/static/outputs', video_name)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # Sort the folders by creation time and get the most recent one
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None
    # Check if there is a most recent run folder
    if most_recent_run:
        # Construct the path to the counts file within the most recent run folder
        line_crossings_file_path = os.path.join(most_recent_run, f'{video_name}_line_crossings.txt')
        # Check if the counts file exists
        if os.path.exists(line_crossings_file_path):
            return line_crossings_file_path
        else:
            return "Line crossings file not found in the most recent run", 404
    else:
        return "No runs found for the video", 404
    

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOADED_VIDEOS_DEST'], filename)

@app.route('/coordinates', methods=['POST'])
def coordinates():
    line = request.form.get('line')
    if line:
        with open('backend/coordinates.txt', 'a') as f:
            f.write(line + '\n')
            app.logger.debug("save coords")
        return jsonify({'message': 'Coordinates recorded'})
    return jsonify({'message': 'No coordinates provided'}), 400

@app.route('/clear_lines', methods=['POST'])
def clear_lines():
    if exists('backend/coordinates.txt'):
        remove('backend/coordinates.txt')
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


    app.logger.debug(f"Received filename:{filename}")
    app.logger.debug(f"Save video option:{save_video}")

    if filename:
        # Pass both filename and save_video to your processing function
        socketio.start_background_task(process_video, filename, save_video)
        return jsonify({'message': 'Video processing started'}), 200
    return jsonify({'message': 'No video file provided'}), 400

def save_processing_status(filename, status):
    status_file_path = os.path.join('backend/static/processing_status', f'{filename}_status.json')
    os.makedirs(os.path.dirname(status_file_path), exist_ok=True)
    with open(status_file_path, 'w') as status_file:
        json.dump({'status': status}, status_file)

def process_video(filename, save_video):
    save_processing_status(filename, 'processing')
    # Call your track.py script here
    app.logger.debug("Start processing video")
    tracker_base_path = "backend/tracking"
    script_path = f"{tracker_base_path}/track.py"
    output_path = "backend/static/outputs/"
    model_path = f"{tracker_base_path}/models/best.onnx"
    cc_source_path = f"{tracker_base_path}/reference-image-test.jpg"
    day_night_path = "static/day_night.csv"
    # model_path = "../tracking/models/yolov8s-2024-02-14-best_fp16_trt.engine"

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
                    "-f", "backend/coordinates.txt",
                    "-c", model_path,
                    "--save-frames",
                    # "--color_calib_enable",
                    "--color_source_path", cc_source_path,
                    "--color_calib_device", "cpu",
                    "--device", "cpu",
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
                    "-f", "backend/coordinates.txt",
                    "-c", model_path,
                    # "--color_calib_enable",
                    "--color_source_path", cc_source_path,
                    "--color_calib_device", "cpu",
                    "--device", "cpu",
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
                    app.logger.debug(f"Progress: {progress}")
            except FileNotFoundError:
                app.logger.debug("File not found")
                pass  # File not found, continue waiting
            except ValueError:
                app.logger.debug("No valid  progress num")
                pass


            time.sleep(1)  # Wait for a short period before checking again
            processing_seconds += 1
        if os.path.exists('progress.txt'):
            os.remove('progress.txt')
        # stderr = process.stderr.read().decode('utf-8')
        # if stderr:
        #     app.logger.debug(f"Error: {stderr}")
        # After the subprocess has finished and the progress file has been deleted
        socketio.emit('video_processed', {'filename': filename})   
        save_processing_status(filename, 'finished')
    except Exception as e:
        app.logger.debug(f"Error: {e}")
        save_processing_status(filename, 'error')

@app.route('/processing_status/<filename>')
def processing_status(filename):
    status_file_path = os.path.join('backend/static/processing_status', f'{filename}_status.json')
    if os.path.exists(status_file_path):
        with open(status_file_path, 'r') as status_file:
            status_data = json.load(status_file)
            return jsonify(status_data)
    else:
        return jsonify({'status': 'unknown'})
    
@app.route('/reprocess', methods=['POST'])
def reprocess():
    # data = request.get_json()
    # app.logger.debug(f"Received data:{data}")  # Add this line to log the received data
    # filename = data.get('filename')
    filename = os.path.split(file_paths[0])[1]
    if filename:
        socketio.start_background_task(reprocess_video, filename)
        return jsonify({'message': 'Video re-processing started'}),  200
    return jsonify({'message': 'No video file provided'}),  400

def reprocess_video(filename):
    # Run the different script on the output_tracks.csv file\
    app.logger.debug(f"reprocess filename {filename}")
    tracker_base_path = "backend/tracking"
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
        app.logger.debug(tracks_input_file)
        if os.path.exists(tracks_input_file):
            try:
                    # Call the reprocess script
                process = subprocess.Popen(['python3', reprocess_script_path,
                                    '--count_lines_file', 'backend/coordinates.txt', 
                                    '--tracks_input_file', tracks_input_file,
                                    '--output_counts_file', output_counts_path,
                                    '--line_crossings', line_crossings_path,
                                    ],
                                    # stdout=subprocess.PIPE,
                                    # stderr=subprocess.PIPE
                                    )
                socketio.emit('progress', {'data': 0})

                processing_seconds = 0

                while process.poll() is None:
                    # app.logger.debug("Waiting for progress")
                    try:
                        with open('progress.txt', 'r') as f:
                            progress = float(f.read().strip())
                            socketio.emit('progress', {'data': progress, 'time': calculate_estimated_time_remaining(progress, processing_seconds)})
                            app.logger.debug(f"Progress: {progress}")
                    except FileNotFoundError:
                        app.logger.debug("Progress file not found")
                        pass  # File not found, continue waiting
                    except ValueError:
                        app.logger.debug("No valid progress num")
                        pass
                    # Read from stdout and stderr
                    # stdout = process.stdout.readline().decode('utf-8')
                    # stderr = process.stderr.readline().decode('utf-8')

                    # # Print to server logs
                    # if stdout:
                    #     app.logger.debug(f"STDOUT: {stdout}")
                    # if stderr:
                    #     app.logger.debug(f"STDERR: {stderr}")

                    time.sleep(1)  # Wait for a short period before checking again
                    processing_seconds += 1
                if os.path.exists('progress.txt'):
                    # Progress file exists, emit 100% progress and delete the file
                    socketio.emit('progress', {'data': 100})
                    os.remove('progress.txt')
                # stderr = process.stderr.read().decode('utf-8')
                # if stderr:
                #     app.logger.debug(f"Error: {stderr}")
                # After the subprocess has finished and the progress file has been deleted
                socketio.emit('video_processed', {'filename': filename})   
            except Exception as e:
                app.logger.debug(f"Error: {e}")
        else:  
            app.logger.debug("Tracks file not found in the most recent run")


@app.route('/get_counts', methods=['GET'])
def get_counts():
    global counts_fig
    filename = os.path.split(file_paths[0])[1]
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('backend/static/outputs', video_name)
    app.logger.debug(base_path)
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
            

            # Assuming data is a dictionary with 'counts' and 'filename' keys
            counts_string = counts_data
            lines = counts_string.strip().split('\n\n')
            counts_data = []
            current_line = 0
            avg_fps = None

            for line in lines:
                parts = line.split('\n')
                line_name = parts[0]
                app.logger.debug(f'Line: {line_name}')
                app.logger.debug(f'Parts: {parts}')

                for part in parts:
                    app.logger.debug(f'Part: {part}')
                    class_parts = re.split(r'[\s,]+', part)  # This splits on whitespace, which should work similarly to the regex used in JS
                    app.logger.debug(f'Class parts: {class_parts}')
                    if class_parts[0] == 'line':
                        current_line = class_parts[1]
                    elif class_parts[0] == 'Average' and class_parts[1] == 'FPS:':
                        avg_fps = class_parts[2]
                    else:
                        class_data = class_parts[0].split('_')
                        counts_data.append({
                            'line': f'Line {current_line}',
                            'class': class_data[0],
                            'direction': class_data[1],
                            'count': class_parts[1]
                        })

            # Creating the data for the Plotly graph
            graph_data = []

            for count in counts_data:
                trace_name = f'Class {count["class"]} {count["direction"]}'
                trace = next((trace for trace in graph_data if trace['name'] == trace_name), None)
                if not trace:
                    trace = {
                        'x': [],
                        'y': [],
                        'type': 'bar',
                        'name': trace_name
                    }
                    # trace['x'].append(count['line'])
                    # trace['y'].append(count['count'])
                    graph_data.append(trace)
                
                trace['x'].append(count['line'])
                trace['y'].append(int(count['count']))
                app.logger.debug(trace)

            # Creating the layout for the Plotly graph
            layout = {
                'title': 'Counts by Line',
                'xaxis': {'title': 'Line'},
                'yaxis': {'title': 'Count'},
                'barmode': 'group'
            }
            app.logger.debug(f"graph data: {graph_data}")
            # Creating the Plotly graph
            counts_fig = go.Figure(data=[go.Bar(name=trace['name'], x=trace['x'], y=trace['y']) for trace in graph_data], layout=layout)
            counts_fig.update_layout(title='Counts by Line', xaxis_title='Line', yaxis_title='Count', barmode='group')

            # Return the plot as a json
            fig_json = counts_fig.to_json()

            return jsonify({'plot': fig_json, 'countsData': counts_data, 'filename': str(video_name)})

            # # Return the counts data as a JSON response
            # return jsonify({'counts': str(counts_data), 'filename': str(video_name)})
        else:
            return jsonify({'message': 'Counts file not found in the most recent run'}),  404
    else:
        return jsonify({'message': 'No runs found for the video'}),  404

@app.route('/get_crossings_data', methods=['POST'])
def get_crossings_data():
    global crossings_fig
    filename = os.path.split(file_paths[0])[1]
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('backend/static/outputs', video_name)
    app.logger.debug(base_path)
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
        # plotly_data = hourly_counts.to_json(orient='records', date_format='iso')


        # Transform data into Plotly format
        transformed_data = {}
        for d in hourly_counts.to_dict('records'):
            # Create a unique key for each combination
            key = f"{d['line_num']}_{d['class_name']}_{d['direction']}"
            if key not in transformed_data:
                transformed_data[key] = {
                    'x': [],
                    'y': [],
                    'name': f"Line {d['line_num']} {d['class_name']} ({d['direction']})",
                    'type': 'bar'
                }
            # Assuming d['time'] is already a date string or a Date object compatible with Plotly
            transformed_data[key]['x'].append(d['time'].isoformat())  # Convert time to string if not already
            transformed_data[key]['y'].append(d['count'])

        # Create traces from the transformed data
        plot_data = list(transformed_data.values())

        # Configure the layout
        layout = {
            'barmode': 'group',  # or 'stack' for stacked bars
            'title': 'Counts per 15 min',
            'xaxis': {
                'title': 'Time',
                'tickangle': -45
            },
            'yaxis': {
                'title': 'Count'
            },
            'margin': {'b': 150}  # Adjust the bottom margin to prevent labels from being cut off
        }

        # Render the Plotly plot
        crossings_fig = go.Figure(data=plot_data, layout=layout)
        crossings_fig.update_layout(
            barmode='group',
            title='Counts per 15 min',
            xaxis=dict(title='Time', tickangle=-45),
            yaxis=dict(title='Count'),
            margin=dict(b=150)
        )
        
        fig_json = crossings_fig.to_json()

        # return jsonify(plotly_data)
        return jsonify({'plot': fig_json, 'filename': str(video_name)})

    else:
        return jsonify({'message': 'No runs found for the video'}),  404


@app.route("/get_person_volume_data", methods=['POST'])
def get_person_volume():
    global volume_fig
    filename = os.path.split(file_paths[0])[1]
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('backend/static/outputs', video_name)
    app.logger.debug(base_path)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # Sort the folders by creation time and get the most recent one
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None
    # Check if there is a most recent run folder
    if most_recent_run:
        data = request.json
        fps = data['fps']  # Get fps from the request data

        # Read the line_crossings.txt file and parse the data
        person_volume_file_path = os.path.join(most_recent_run, f'{video_name}_person_volume.txt')
        volume_df = pd.read_csv(person_volume_file_path, header=None, skiprows=1, sep=',')

        volume_df.columns = ['frame_num', 'volume']

        # Calculate timestamps and aggregate by hour
        volume_df['timestamp'] = pd.to_datetime(volume_df['frame_num'] / fps, unit='s')

        # Format timestamps to only include the time portion
        volume_df['time'] = volume_df['timestamp'].dt.time

        # volume_df.set_index('time', inplace=True)
        # hourly_counts = volume_df.groupby([pd.Grouper(freq='15min')]).size().reset_index(name='count')

        # Use the formatted 'time' for plotting
        # volume_df['time'] = volume_df['timestamp'].dt.time
        volume_df.drop(columns=['timestamp'], inplace=True)

        # smoothing
        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size

        volume_df['volume'] = np.convolve(volume_df['volume'], kernel, mode='same')

        volume_fig = go.Figure()

        volume_fig.add_trace(go.Scatter(x=volume_df['time'], y=volume_df['volume'], mode='lines', name='Person Volume'))

        volume_fig.update_layout(title="Volume of People Over Time",
                xaxis=dict(showgrid=True), yaxis=dict(showgrid=True),
                margin=dict(b=150))

        fig_json = volume_fig.to_json()

        # return jsonify(plotly_data)
        return jsonify({'plot': fig_json, 'filename': str(video_name)})

    else:
        return jsonify({'message': 'No runs found for the video'}),  404


def get_tracks(file_name):
    """"Use it to get the tracks into a dictionary, then calc the counts over the lines."""
    track_df = pd.read_csv(file_name, header=None)

    frame_index = 1
    # train_labels_1 is a pandas dataframe with columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, -1, -1
    track_df = track_df.sort_values([1, 0]) # sort by id first then frame second
    app.logger.debug(track_df.head())

    # Initialize an empty dictionary
    bounding_boxes = {}

    # Group the DataFrame by 'track_id' and iterate over each group
    for track_id, group in track_df.groupby(1):
        # Create a list of tuples for the bounding box coordinates
        # put into numpy array
        # app.logger.debug(group)
        boxes = np.array(list(zip(group[2], group[3], group[4], group[5], group[7])))
        class_ids = np.array(list(group[7]))

        # Assign this list to the corresponding track_id in the dictionary
        bounding_boxes[track_id] = boxes

    # Display the resulting dictionary
    # app.logger.debug(bounding_boxes)
    return bounding_boxes

def save_middle_video_frame(file_path, save_path):
    """Takes a video, extracts the middle frame and saves it into save_path
    Also returns the video resolution"""
    # Open the video file
    cap = cv.VideoCapture(file_path)

    if not cap.isOpened():
        app.logger.debug("Could not open video. Looking for middle_frame.jpg")
        try:
            img = cv.imread(save_path, cv.IMREAD_COLOR)
            height, width, channels = img.shape
            resolution = (width, height)
            return resolution
        except Exception as e:
            app.logger.debug("Could not read middle_frame.jpg")
            app.logger.debug(f"Error: {e}")
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
            app.logger.debug("Error: Could not read the middle frame.")
        
        # Release the video capture object
        cap.release()
    return resolution

def parse_count_lines_file(count_lines_file):
  '''File format:
  (x1,y1) (x2,y2)
  (x1,y1) (x2,y2)'''
  with open(count_lines_file, 'r') as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  lines = [line.split(' ') for line in lines]
  lines = [[eval(coord.replace('(', '').replace(')', '')) for coord in line] for line in lines]
  return lines

# Function to determine the direction and positioning of the arrow based on the line
def add_line_with_annotations(fig, line, line_index):
    start_x, start_y = line[0]
    end_x, end_y = line[1]
    
    # Calculate vector from start to end
    vec_x = end_x - start_x
    vec_y = end_y - start_y

    # Normalize the vector
    norm = (vec_x ** 2 + vec_y ** 2) ** 0.5
    unit_vec_x = vec_x / norm
    unit_vec_y = vec_y / norm

    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2

    # Parameters for arrow size
    arrow_length = 50  # Length of the arrow

    # Compute arrow vectors (opposite direction for "In")
    arrow_vec_x_in = unit_vec_y * arrow_length
    arrow_vec_y_in = -unit_vec_x * arrow_length

    arrow_vec_x_out = -unit_vec_y * arrow_length
    arrow_vec_y_out = unit_vec_x * arrow_length

    # Define offsets for "In" and "Out" labels based on the line vector
    offset_distance = 25  # Distance from the line to place labels
    offset_x = unit_vec_y * offset_distance  # Perpendicular to line vector
    offset_y = -unit_vec_x * offset_distance

    label_offset_x = unit_vec_x * offset_distance
    label_offset_y = unit_vec_y * offset_distance

    # Add the line as a shape
    fig.add_shape(type="line", x0=start_x, y0=start_y, x1=end_x, y1=end_y,
                  line=dict(color="RoyalBlue", width=3))
    
    # Add "In" arrow annotation at the start
    fig.add_annotation(x=mid_x - offset_x, y=mid_y - offset_y, ax=arrow_vec_x_in, ay=arrow_vec_y_in,
                       text="In", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                       arrowcolor="Blue", font=dict(size=15, color="Blue"))

    # Add "Out" arrow annotation at the end
    fig.add_annotation(x=mid_x + offset_x + label_offset_x, y=mid_y + offset_y + label_offset_y, ax=arrow_vec_x_out, ay=arrow_vec_y_out,
                       text="Out", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                       arrowcolor="DarkGreen", font=dict(size=15, color="DarkGreen"))

    # Add a label for the line's name near its midpoint

    fig.add_annotation(x=end_x, y=end_y,
                       text=line_index, # or f"Line {line_index}" for a generic label
                       showarrow=False, font=dict(size=15, color="Red"),
                       bgcolor="white", bordercolor="Red", borderwidth=2, borderpad=4)


@app.route('/get_tracks', methods=['POST'])
def get_track_data_plot():
    global track_fig
    filename = os.path.split(file_paths[0])[1]
    video_path = file_paths[0]
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('backend/static/outputs', video_name)
    app.logger.debug(base_path)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # Sort the folders by creation time and get the most recent one
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None
    # Check if there is a most recent run folder
    if most_recent_run:
        run_name = os.path.basename(most_recent_run)
        # Read the tracks.txt file and parse the data
        tracks_file_path = os.path.join(most_recent_run, f'{video_name}_tracks_output.txt')
        app.logger.debug(f"get tracks filename: {tracks_file_path}")
        count_lines_file = "backend/coordinates.txt"
        if os.path.exists(count_lines_file):
            count_lines = parse_count_lines_file(count_lines_file)
        if os.path.exists(tracks_file_path):
            # tracks is a dict from track_ids to their array of bboxes in order of frame
            tracks = get_tracks(tracks_file_path)
            # Get the source video path
            # stored in static/uploads/
            # video_path = os.path.join('static/uploads', filename)
            saved_frame = f"{most_recent_run}/middle_frame.jpg"
            resolution = save_middle_video_frame(video_path, saved_frame)
            app.logger.debug(f"saved_frame {saved_frame}")
            saved_frame = f"backend/static/outputs/{video_name}/{run_name}/middle_frame.jpg"
            # saved_frame = os.path.join(os.path.curdir, saved_frame)
            app.logger.debug(f"saved_frame frontend {saved_frame}")
            kernel_size = 10
            kernel = np.ones(kernel_size) / kernel_size

            # Create subplots
            track_fig = make_subplots(rows=1, cols=1, subplot_titles=("10 frame smoothing"))

            color_dict = {0: "Yellow", 1: "Red", 2: "Green", 3: "Blue"}

            # traces = [trace for trace in tracks.values() if len(trace) > 4]

            # # Collect all end points from tracks
            # end_points = np.array([
            #     (trace[-1, 0] + trace[-1, 2] // 2, trace[-1, 1] + trace[-1, 3]) 
            #     for trace in traces
            # ])

            # # max distance between two points in a cluster
            # my_epsilon = 100
            # my_min_samples = 2


            # # Run DBSCAN clustering on the collected end points
            # db = DBSCAN(eps=my_epsilon, min_samples=my_min_samples).fit(end_points)
            # labels = db.labels_
            # app.logger.debug(f"db scan labels: {labels}")

            # # Create a dictionary to hold lines for each cluster
            # clustered_lines = {label: [] for label in set(labels) if label != -1}

            # # The labels array now holds the cluster ID for each point in all_points_array
            # # Points with the same label belong to the same cluster
            # # Points with the label -1 are considered noise and not part of any cluster

            # # Add lines to their respective clusters based on labels
            # for label, trace in zip(labels, traces):
            #     if label != -1:
            #         clustered_lines[label].append(trace)


            # # Draw a smooth line for each cluster
            # for cluster_id, lines in clustered_lines.items():
            #     if not lines:  # Skip if there are no lines in the cluster
            #         continue

            #     # Collect all points from the traces in the cluster
            #     all_x_points = []
            #     all_y_points = []
            #     for line in lines:
            #         all_x_points.extend(line[:, 0] + line[:, 2] // 2)
            #         all_y_points.extend(line[:, 1] + line[:, 3])

            #     # # Interpolate a spline through the points
            #     # tck, u = splprep([all_x_points, all_y_points], s=0)
            #     # new_points = splev(u, tck)

            #     # Add trace for the cluster
            #     track_fig.add_trace(go.Scatter(
            #         x=all_x_points,
            #         y=all_y_points,
            #         mode='lines',
            #         line=dict(width=3),  # Adjust line thickness
            #         name=f'Cluster {cluster_id}'
            #     ))

            max_tracks = 200

            if len(tracks.keys()) > max_tracks:
                step_size = math.floor(len(tracks.keys()) // max_tracks)
            else:
                step_size = 1

            traces = [trace for trace in tracks.values() if len(trace) > 20]

            
            for i, trace in enumerate(traces):
                # if i % step_size == 0:
                # class_ids = tracks[track_id][1]
                # class_id = np.argmax(np.histogram(class_ids, [0, 1, 2, 3])[0])
                if len(trace) != 0:
                    # annotate the center of the bottom line of the BBox, because that is intuitive
                    trace_center_x = trace[:,0] + trace[:,2] // 2
                    trace_bottom_y = trace[:,1] + trace[:,3]
                    trace_convolved_x = np.convolve(trace_center_x, kernel, mode='valid')
                    trace_convolved_y = np.convolve(trace_bottom_y, kernel, mode='valid')
                    # Flip the y coords because images are index top to bottom
                    trace_convolved_y = trace_convolved_y

                    # Add smoothed path
                    track_fig.add_trace(
                        go.Scatter(
                            x=trace_convolved_x,
                            y=trace_convolved_y,
                            mode='lines',
                            name=f'Trace {i} smoothed',
                            opacity=0.2,
                            marker=dict(
                                color='red',
                            ),
                        ),
                        row=1, col=1
                    )
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
            track_fig.update_xaxes(title_text="X", range=[0, resolution[0]], row=1, col=1)
            track_fig.update_yaxes(title_text="Y", range=[resolution[1], 0], row=1, col=1)

            # aspect ratio of image
            # aspect_ratio = resolution[0]/resolution[1]
            # plot_width = 800
            # plot_height = int(plot_width / aspect_ratio)
            # Set a background image
            # track_fig.update_layout(
            #     images=[
            #         dict(
            #             source=saved_frame,
            #             xref="x", yref="y",  # Use "paper" to refer to the whole plotting area
            #             x=0, y=1,  # These specify the position of the image (0,0 is bottom left, 1,1 is top right)
            #             sizex=resolution[0], sizey=resolution[1],  # These specify the size of the image. 1,1 will cover the entire background
            #             sizing="stretch",
            #             opacity=1.0,  # Set the opacity of the image
            #             layer="below"  # Ensure the image is below the plot
            #         )
            #     ]
            # )
            # Set the layout to include a background image
            track_fig.update_layout(
                images=[go.layout.Image(
                    source=Image.open(saved_frame),  # Path to your background image
                    xref="x", yref="y", 
                    x=0, y=1,
                    sizex=resolution[0], sizey=resolution[1],  # These specify the size of the image. 1,1 will cover the entire background
                    sizing="stretch",
                    opacity=1.0,  # Adjust opacity if needed
                    layer="below")])

            start_ind = 0
            end_ind = 1

            # Add lines and annotations
            for i,line in enumerate(count_lines):
                add_line_with_annotations(track_fig, line, i)

            # Set the layout of the figure
            track_fig.update_layout(title="Lines with Arrows and Labels",
                            xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))

            # Update layout if needed
            track_fig.update_layout(height=resolution[1], width=resolution[0], title_text="Path Analysis", title_x=0.5)

            fig_json = track_fig.to_json()
            return jsonify({'fig_json': fig_json, 'filename': video_name})
        else:
            return jsonify({'message': 'No tracks found for this video'}), 404
    else:
        return jsonify({'message': 'No runs found for the video'}),  404
    
def save_plots_as_image(figure, save_path):
    figure.write_image(save_path)

@app.route('/get_plots')
def get_plots():
    global track_fig, counts_fig, crossings_fig, volume_fig
    figures = [track_fig, counts_fig, crossings_fig, volume_fig]
    filename = os.path.split(file_paths[0])[1]
    video_path = file_paths[0]
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('backend/static/outputs', video_name)
    app.logger.debug(base_path)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    # Sort the folders by creation time and get the most recent one
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None
    # Check if there is a most recent run folder
    if most_recent_run:
        # Read the tracks.txt file and parse the data
        plots_dir_path = os.path.join(most_recent_run, f'{video_name}_plots')
        if not os.path.exists(plots_dir_path):
            os.mkdir(plots_dir_path)
        tracks_image_path = os.path.join(plots_dir_path, f'{video_name}_tracks_overlay.png')
        counts_image_path = os.path.join(plots_dir_path, f'{video_name}_counts.png')
        crossings_image_path = os.path.join(plots_dir_path, f'{video_name}_15_min.png')
        volume_image_path = os.path.join(plots_dir_path, f'{video_name}_person_volume.png')
        save_paths = [tracks_image_path, counts_image_path, crossings_image_path, volume_image_path]

        num_errors = 0

        for i in range(len(figures)):
            try:
                save_plots_as_image(figures[i], save_paths[i])
            except Exception as err:
                app.logger.debug(f"figure {save_paths[i]} not found")
                app.logger.debug(f"Error: {err}")
                num_errors += 1
        socketio.emit('plot-download-ready')
        if num_errors == len(figures):
            return "No figures found", 404
        return plots_dir_path
    else:
        return "No runs found for the video", 404



def shutdown_handler():
    """Handle shutdown. Delete coordinates file"""
    if exists('backend/coordinates.txt'):
        remove('backend/coordinates.txt')

if __name__ == '__main__':
    # Adjust logging level as necessary
    app.logger.setLevel(logging.DEBUG)
    app.logger.debug("start webapp")
    atexit.register(shutdown_handler)
    try:
        app.run(debug=False, port=5000)
        
    except KeyboardInterrupt:
        shutdown_handler()
    finally:
        app.logger.debug("Flask server has shut down.")
