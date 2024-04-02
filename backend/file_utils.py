import os
from os import remove
from os.path import exists
import glob

def find_most_recent_run(video_name: str):
    '''
        This function finds the path to the most recent output run
        for the uploaded video.
        video_name (str): The name of the video file.
        
        Returns the path to the most recent output run folder.
    '''

    video_name = video_name.split('.')[0]

    base_path = os.path.join('backend/static/outputs', video_name)
    run_folders = glob.glob(os.path.join(base_path, 'run_*'))
    run_folders.sort(key=os.path.getctime, reverse=True)
    most_recent_run = run_folders[0] if run_folders else None

    return most_recent_run