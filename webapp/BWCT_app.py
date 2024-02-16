import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask import send_from_directory

from flask_uploads import UploadSet, configure_uploads, ALL
from flask import Response, stream_with_context
from flask_socketio import SocketIO

from werkzeug.utils import secure_filename
import os
import subprocess
from os import remove
from os.path import exists
import glob

app = Flask(__name__)
socketio = SocketIO(app)
# Secret flash messaging
app.secret_key = "supersecretkey"

videos = UploadSet('videos', ALL)

app.config['UPLOADED_VIDEOS_DEST'] = 'static/uploads'
configure_uploads(app, videos)


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        video = request.files.get('video')
        if video:
            filename = secure_filename(video.filename)
            video.save(os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename))
            if exists('coordinates.txt'):
                remove('coordinates.txt')
            return jsonify({'message': 'Video uploaded', 'filename': filename})
        return jsonify({'message': 'No video provided'}), 400
    else:
        return render_template('upload.html')  # replace 'index.html' with your actual template


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


@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    print("Received data:", data)  # Add this line to log the received data
    filename = data.get('filename')
    if filename:
        socketio.start_background_task(process_video, filename)
        return jsonify({'message': 'Video processing started'}),  200
    return jsonify({'message': 'No video file provided'}),  400

def process_video(filename):
    # Call your track.py script here
    print("Start processing video")
    script_path = "../Impr-Assoc-counter_demo/track.py"
    output_path = "static/outputs/"
    model_path = "../Impr-Assoc-counter_demo/models/yolov8s-2024-02-14-best_fp16_trt.engine"
    video_path = os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename)
    try:
        process = subprocess.Popen(
            [
                "python", script_path,
                "--source_video_path", video_path,
                "--output_dir", output_path,
                "-f", "coordinates.txt",
                "-c", model_path
            ],
             stdout=subprocess.PIPE,
             stderr=subprocess.PIPE)
        socketio.emit('progress', {'data': 0})

        while process.poll() is None:
            try:
                with open('progress.txt', 'r') as f:
                    progress = float(f.read().strip())
                    socketio.emit('progress', {'data': progress})
                    print(f"Progress: {progress}")
            except FileNotFoundError:
                print("File not found")
                pass  # File not found, continue waiting

            time.sleep(1)  # Wait for a short period before checking again
        if os.path.exists('progress.txt'):
            os.remove('progress.txt')
        stderr = process.stderr.read().decode('utf-8')
        if stderr:
            print(f"Error: {stderr}")
        # After the subprocess has finished and the progress file has been deleted
        socketio.emit('video_processed', {'filename': filename})   
    except Exception as e:
        print(f"Error: {e}")
    
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

if __name__ == '__main__':
    app.run(debug=True)