import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask import send_from_directory, send_file

from flask_uploads import UploadSet, configure_uploads, ALL
from flask import Response, stream_with_context
from flask_socketio import SocketIO

from werkzeug.utils import secure_filename
import os
import subprocess
from os import remove
from os.path import exists
import glob

import pandas as pd
import math

app = Flask(__name__)
socketio = SocketIO(app)
# Secret flash messaging
app.secret_key = "supersecretkey"

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
            filename = secure_filename(video.filename)
            video_path = os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename)
            video.save(video_path)
            # os.system(f"ffmpeg -i {video_path} -c:v libx264 -preset veryfast -crf 23 {app.config['UPLOADED_VIDEOS_DEST']}/{filename.split('.')[0]}.mp4")
            if exists('coordinates.txt'):
                remove('coordinates.txt')
            return jsonify({'message': 'Video uploaded', 'filename': filename})
        return jsonify({'message': 'No video provided'}), 400
    else:
        return render_template('upload.html')  # replace 'index.html' with your actual template
    
@app.route('/stream_video')
def stream_video():
    filename = request.args.get('filename')  # Get filename from query parameter
    if not filename:
        return "Filename not provided", 400
    filename = os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename)

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

@app.route('/download_counts/<filename>')
def download_counts(filename):
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

@app.route('/download_processed_video/<filename>')
def download_video(filename):
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
    filename = request.form.get('filename')
    save_video = request.form.get('save_video')
    if save_video == "yes":
        save_video = True
    elif save_video == "no":
        save_video = False

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
    script_path = "../Impr-Assoc-counter_demo/track.py"
    output_path = "static/outputs/"
    model_path = "../Impr-Assoc-counter_demo/models/yolov8s-2024-02-16-best_fp16_trt.engine"
    cc_source_path = "../Impr-Assoc-counter_demo/reference-image-test.jpg"
    # model_path = "../Impr-Assoc-counter_demo/models/yolov8s-2024-02-14-best_fp16_trt.engine"

    video_path = os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename)
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
                    # "--color_source_path", cc_source_path
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
                    # "--color_source_path", cc_source_path
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
    data = request.get_json()
    print("Received data:", data)  # Add this line to log the received data
    filename = data.get('filename')
    if filename:
        socketio.start_background_task(reprocess_video, filename)
        return jsonify({'message': 'Video re-processing started'}),  200
    return jsonify({'message': 'No video file provided'}),  400

def reprocess_video(filename):
    # Run the different script on the output_tracks.csv file\
    reprocess_script_path = "../Impr-Assoc-counter_demo/reprocess_tracks.py"
    base_path = "static/outputs/"
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



@app.route('/counts/<filename>', methods=['GET'])
def get_counts(filename):
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('static/outputs', video_name)
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
            return jsonify({'counts': str(counts_data)})
        else:
            return jsonify({'message': 'Counts file not found in the most recent run'}),  404
    else:
        return jsonify({'message': 'No runs found for the video'}),  404

@app.route('/get_crossings_data/<filename>', methods=['POST'])
def get_crossings_data(filename):
    # Construct the path to the counts file
    video_name = filename.split('.')[0]
    base_path = os.path.join('static/outputs', video_name)
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
        hourly_counts = crossings_df.groupby([pd.Grouper(freq='H'), 'line_num', 'class_name', 'direction']).size().reset_index(name='count')

        # Use the formatted 'time' for plotting
        hourly_counts['time'] = hourly_counts['timestamp'].dt.time
        hourly_counts.drop(columns=['timestamp'], inplace=True)

        # Convert to JSON or a suitable format for the frontend
        plotly_data = hourly_counts.to_json(orient='records', date_format='iso')
        
        return jsonify(plotly_data)

    else:
        return jsonify({'message': 'No runs found for the video'}),  404

if __name__ == '__main__':
    app.run(debug=False)