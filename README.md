<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" alt="project-logo">
</p>
<p align="center">
    <h1 align="center"></h1>
</p>
<p align="center">
    <em>Using Computer Vision to count and analyze how vulnerable road users use streets so that local governments have easy access to data for justifying investment in pedestrian and bike infrastructure.</em>
</p>
<p align="center">
	<!-- local repository, no metadata badges. -->
<p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=default&logo=tqdm&logoColor=black" alt="tqdm">
	<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=default&logo=TensorFlow&logoColor=white" alt="TensorFlow">
	<img src="https://img.shields.io/badge/JavaScript-F7DF1E.svg?style=default&logo=JavaScript&logoColor=black" alt="JavaScript">
	<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=default&logo=scikit-learn&logoColor=white" alt="scikitlearn">
	<img src="https://img.shields.io/badge/HTML5-E34F26.svg?style=default&logo=HTML5&logoColor=white" alt="HTML5">
	<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=default&logo=YAML&logoColor=white" alt="YAML">
	<img src="https://img.shields.io/badge/C-A8B9CC.svg?style=default&logo=C&logoColor=black" alt="C">
	<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=default&logo=SciPy&logoColor=white" alt="SciPy">
	<img src="https://img.shields.io/badge/Electron-47848F.svg?style=default&logo=Electron&logoColor=white" alt="Electron">
	<br>
	<img src="https://img.shields.io/badge/Plotly-3F4F75.svg?style=default&logo=Plotly&logoColor=white" alt="Plotly">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=default&logo=Docker&logoColor=white" alt="Docker">
	<img src="https://img.shields.io/badge/GitHub%20Actions-2088FF.svg?style=default&logo=GitHub-Actions&logoColor=white" alt="GitHub%20Actions">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/ONNX-005CED.svg?style=default&logo=ONNX&logoColor=white" alt="ONNX">
	<img src="https://img.shields.io/badge/JSON-000000.svg?style=default&logo=JSON&logoColor=white" alt="JSON">
	<img src="https://img.shields.io/badge/Flask-000000.svg?style=default&logo=Flask&logoColor=white" alt="Flask">
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#-overview)
- [ Features](#-features)
- [ Repository Structure](#-repository-structure)
- [ Modules](#-modules)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Tests](#-tests)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)
</details>
<hr>

## Overview

This repository contains code for an Electron app that processes videos from traffic cameras or portable cameras to count the number of people passing through the video. The user can draw lines on the video to indicate where they want to count people passing by. The app can distinguish between pedestrians, bikes, electric scooter riders, and wheelchairs (coming soon). It provides a frontend Electron interface that connects to a Flask backend to handle the video processing and people counting logic. The app ensures connectivity between the frontend and backend, monitors the backend health, handles reconnections as needed, and enables saving output files. It is designed to provide a seamless user experience for processing videos to count people crossing designated areas.  

You can see our report [here](https://docs.google.com/document/d/1ou7F_Dk361bpQnsbr4r7yQ6nwmXW1UgSbhNV0eN8LDI/edit?usp=sharing).


##  Features

|    | Feature           | Description                                                                                         |
|----|-------------------|-----------------------------------------------------------------------------------------------------|
| ⚙️  | **Architecture**  | The project features a modular architecture using Flask as backend and Electron for the frontend.   |
| 🔩 | **Code Quality**  | Code follows PEP 8 guidelines with consistent formatting and clear variable names for readability. |
| 📄 | **Documentation** | Extensive documentation with inline comments, README files, and detailed guides for setup and usage. |
| 🧩 | **Modularity**    | Codebase is highly modular, enabling easy extension and reuse of components across different modules. |
| ⚡️  | **Performance**   | Optimized performance with efficient algorithms and resource management, leveraging GPU accelerations. |
| 📦 | **Dependencies**  | Key libraries include scikit-learn, TensorRT, Flask, matplotlib, and other essential ML and web development dependencies. |
| 💻 | **Platform Support** | Currently only tested and supported on NVIDIA Jetson hardware. |

`

---

##  Repository Structure

```sh
└── /
    ├── BWCT_favicon.png
    ├── README.md
    ├── backend
    │   ├── BWCT_app.py
    │   ├── tracking
    │   │   ├── ConfTrack
    │   │   │   ├── basetrack.py
    │   │   │   ├── ConfTrack.py
    │   │   │   ├── kalman_filter.py
    │   │   │   └── matching.py
    │   │   ├── Impr_Assoc_Track
    │   │   │   ├── basetrack.py
    │   │   │   ├── Impr_Assoc_Track.py
    │   │   │   ├── interpolation.py
    │   │   │   ├── kalman_filter.py
    │   │   │   └── matching.py
    │   │   ├── LSTMTrack
    │   │   │   ├── LSTM_predictor.py
    │   │   │   └── LSTMTrack.py
    │   │   ├── YOLOv8_TensorRT
    │   │   ├── color_transfer_cpu.py
    │   │   ├── color_transfer_gpu.py
    │   │   ├── example_count_lines.txt
    │   │   ├── reprocess_tracks.py
    │   │   ├── requirements.txt
    │   │   ├── setup.py
    │   │   └── track.py
    │   └── templates
    ├── index.html
    ├── main.js
    ├── package-lock.json
    ├── package.json
    ├── preload.js
    ├── readme-ai.md
    ├── requirements.txt
    ├── test_main.js
    └── track_logs.txt
```

---

##  Modules

<details closed><summary>.</summary>

| File                                   | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---                                    | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [test_main.js](test_main.js)           | Creates and manages a responsive Electron app window connected to a Flask backend. Monitors backend health, reconnects if needed, and facilitates file saving via dialogs. Initiated on app launch and handles system sleep events for seamless operation.                                                                                                                                                                                                                                                                                                             |
| [package-lock.json](package-lock.json) | Code SummaryThe code file `BWCT_app.py` in the `backend` directory serves as the core application logic for the parent repository. This file is the backbone of the web application, handling various backend functionalities such as routing, data processing, and communication with the frontend. It orchestrates the interactions between different components, ensuring a seamless flow of data and actions within the application. Additionally, it encapsulates critical business logic and serves as the central hub for managing user requests and responses. |
| [track_logs.txt](track_logs.txt)       | The `track_logs.txt` file in the repository serves as a log for the processing of frames in the project. It records the progress of processing each frame out of a total of 1443 frames, including the frame number and frames per second (fps) at that point. This log file provides valuable insights into the processing speed and progress of the project execution, aiding in monitoring and optimizing performance.                                                                                                                                              |
| [requirements.txt](requirements.txt)   | NumPy for array manipulation-OpenCV for computer vision tasks-Flask for web framework-TensorFlow for machine learning-Plotly for interactive visualizations-Pandas for data analysis-And more crucial libraries.                                                                                                                                                                                                                                                                                                                                                       |
| [preload.js](preload.js)               | Enables secure communication between front-end and back-end in Electron app. Exposes functions to interact with file system such as opening files, saving raw data, and generating visualizations. Enhances user experience by facilitating file operations seamlessly.                                                                                                                                                                                                                                                                                                |
| [package.json](package.json)           | Defines metadata for an Electron app named bwct-tracker-electron within the repository. Specifies dependencies, scripts for app execution, author details, licensing, and repository links, crucial for managing the Electron app within the project architecture.                                                                                                                                                                                                                                                                                                     |
| [main.js](main.js)                     | Integrates Node.js with Python backend for enhanced functionality. Contributes to seamless operation of the hybrid application within the repositorys architecture.                                                                                                                                                                                                                                                                                                                                                                                                    |
| [index.html](index.html)               | Enables real-time file upload, merging, and status tracking for the BWCT Video Merging Tool web application. Supports multi-file uploads, async merge requests, and live status updates via Resumable.js and server-side endpoints.                                                                                                                                                                                                                                                                                                                                    |

</details>

<details closed><summary>backend</summary>

| File                               | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---                                | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [BWCT_app.py](backend/BWCT_app.py) | The `BWCT_app.py` file in the backend directory of the repository serves as the core Flask application handling file uploads, real-time data visualization, and interaction using SocketIO. It enables users to upload files, process data, and retrieve visualizations. Additionally, it incorporates features for monitoring file changes and serving downloadable content. The file also integrates various libraries for file handling, event observation, and data manipulation to provide a robust platform for user interaction and data analysis. |

</details>

<details closed><summary>backend.templates</summary>

| File                                         | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---                                          | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [upload.html](backend/templates/upload.html) | The `upload.html` file within the `backend/templates` directory of the repository serves as the user interface for the BWCT Video Analysis Tool. It provides a web interface for users to upload videos for analysis. The page includes necessary styling and scripts for functionality, such as handling uploads and displaying analysis results. The primary purpose of this file is to facilitate the seamless uploading of videos and enhance the user experience within the broader architecture of the BWCT application. |

</details>

<details closed><summary>backend.tracking</summary>

| File                                                                               | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---                                                                                | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [requirements.txt](backend/tracking/requirements.txt)               | Improve association and counters with essential libraries for image processing, machine learning, visualization, and data handling.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [reprocess_tracks.py](backend/tracking/reprocess_tracks.py)         | Generates line crossing counts by processing tracks and count lines, utilizing specified input files and saving the results. Parses line and track data to calculate crossings for different object classes, updating the counts accordingly.                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [example_count_lines.txt](backend/tracking/example_count_lines.txt) | Implements line counting functionality for demonstrating improved association with coordinates (640,0) to (640,720) in the tracking. Located at backend/tracking/example_count_lines.txt within the repository structure.                                                                                                                                                                                                                                                                                                                                                                                                             |
| [track.py](backend/tracking/track.py)                               | Backend/tracking/track.py`The `track.py` file within the `tracking` directory focuses on implementing video processing, object detection, and tracking functionalities using various libraries and modules. It leverages libraries like OpenCV, NumPy, TensorFlow, and PyTorch for efficient video analysis. Additionally, it integrates YOLOv8 TensorRT for optimized real-time object detection and tracking. The script aims to provide a robust solution for enhancing association and counting tasks within video analysis applications.                                                                                         |
| [color_transfer_cpu.py](backend/tracking/color_transfer_cpu.py)     | Implements color transfer between images using mean and standard deviations in the Lab color space. Enhances aesthetics by adjusting brightness levels, following original paper methodology or alternative scaling. Outputs a visually appealing color-transferred image.                                                                                                                                                                                                                                                                                                                                                                                          |
| [color_transfer_gpu.py](backend/tracking/color_transfer_gpu.py)     | The `color_transfer_gpu.py` file, located within the `backend/tracking` directory, facilitates GPU-accelerated color transfer operations for enhanced performance. It leverages libraries like OpenCV, NumPy, CuPy, and Torch to efficiently manipulate and transfer image pixel data between various formats. The code defines a function to convert a GPU matrix from OpenCV to a CuPy array, enabling seamless interaction between GPU-accelerated matrices in different libraries. This functionality contributes to optimizing image processing tasks within the parent repositorys architecture, improving overall performance and efficiency. |
| [Impr_track_count.py](backend/tracking/Impr_track_count.py)         | Backend/tracking/Impr_track_count.py`The `Impr_track_count.py` file within the `backend/tracking` directory serves a crucial role in the repositorys architecture by handling image processing and association counting tasks. It leverages various libraries such as matplotlib, scipy, numpy, and TensorFlow to analyze and visualize tracking data efficiently. By integrating functionality from fastreid and other dependencies, this code file supports advanced image processing, machine learning, and data analysis for improved association counting within the project.                                                    |
| [setup.py](backend/tracking/setup.py)                               | Set up compilation environment for PyTorch extension module.-Define extension modules with main source and additional sources.-Retrieve version and long description for the setup.-Configure package details and dependencies for PyPI distribution.                                                                                                                                                                                                                                                                                                                                                                                                               |

</details>

<details closed><summary>backend.tracking.Impr_Assoc_Track</summary>

| File                                                                                        | Summary                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                         | ---                                                                                                                                                                                                                                                                                                                                                                                                             |
| [basetrack.py](backend/tracking/Impr_Assoc_Track/basetrack.py)               | Track state, ID generation, state transitions, position history, activation, prediction, update methods, and state marking. Enables implementing custom tracking logic.                                                                                                                                                                                                                                         |
| [interpolation.py](backend/tracking/Impr_Assoc_Track/interpolation.py)       | Creates interpolation results for tracking data in MOTChallenge format, optimizing track continuity by filling in missing frames through linear interpolation. Parses input arguments, generates new tracklets based on specified thresholds, and writes the interpolated track results to new files.                                                                                                           |
| [Impr_Assoc_Track.py](backend/tracking/Impr_Assoc_Track/Impr_Assoc_Track.py) | This code file within the parent repositorys architecture implements a tracking system that utilizes various algorithms for association and feature extraction. It leverages techniques such as IOU distance calculation, Kalman filtering, and fast re-identification for object tracking. The file integrates these components to provide a robust solution for multi-object tracking in real-time scenarios. |
| [matching.py](backend/tracking/Impr_Assoc_Track/matching.py)                 | Implements functions to calculate matching indices, IoU distances, and cost matrices for multi-object tracking association. Includes fusion methods for motion, IoU, and detection scores. Enables efficient assignment and merging of object matches in multi-object tracking systems.                                                                                                                         |
| [kalman_filter.py](backend/tracking/Impr_Assoc_Track/kalman_filter.py)       | Implements Kalman filtering for tracking bounding boxes with motion model and observation matrix. Facilitates track creation, prediction, correction, and distance computation for state and measurement comparison.                                                                                                                                                                                            |

</details>

<details closed><summary>backend.tracking.ConfTrack</summary>

| File                                                                           | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---                                                                            | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [basetrack.py](backend/tracking/ConfTrack/basetrack.py)         | State transitions, ID generation, activation handling, and state updates. Supports multi-camera tracking with location tracking. Encapsulates key tracking attributes and methods for future extensibility and customization within the parent repositorys architecture.                                                                                                                                                                                                                     |
| [ConfTrack.py](backend/tracking/ConfTrack/ConfTrack.py)         | The `ConfTrack.py` file in the `backend/tracking` directory of the repository implements core functionality for object tracking and association in computer vision tasks. It leverages various matching algorithms, Kalman filtering, and FastReID integration to track objects efficiently. This code plays a crucial role in enhancing the tracking accuracy and robustness of the overall system by utilizing advanced computer vision techniques.                         |
| [ConfTrack](backend/tracking/ConfTrack/ConfTrack)               | Code Summary**This code file within the repositorys backend architecture defines a `STrack` class that serves as a critical component for object tracking functionalities. It imports various modules related to computer vision and tracking algorithms, setting up the necessary infrastructure for object detection, tracking, and feature extraction. The `STrack` class is a key element enabling the system to effectively monitor and maintain the identity of objects across frames. |
| [matching.py](backend/tracking/ConfTrack/matching.py)           | Implements functions to compute distance metrics for object tracking. Utilizes IoU calculations to determine similarity between bounding boxes. Facilitates matching and fusion of tracked objects. Enhances object tracking accuracy through cost optimization.                                                                                                                                                                                                                             |
| [kalman_filter.py](backend/tracking/ConfTrack/kalman_filter.py) | Implements Kalman filtering for tracking bounding boxes in image space. Initializes, predicts, updates state, and computes distance to measurements with customizable metric. Enhances object tracking robustness in ConfTrack architecture.                                                                                                                                                                                                                                                 |

</details>

<details closed><summary>backend.tracking.YOLOv8_TensorRT</summary>

| File                                                                             | Summary                                                                                                                                                                                                                                                                               |
| ---                                                                              | ---                                                                                                                                                                                                                                                                                   |
| [cuda_utils.py](backend/tracking/YOLOv8_TensorRT/cuda_utils.py)   | Transform images on GPUs using CUDA-accelerated functions, resizing and padding while maintaining aspect ratio for compatibility with a new shape.Utilizes PyTorch for GPU padding and seamlessly interfaces with CuPy arrays to convert from OpenCV to PyTorch tensors.              |
| [pycuda_api.py](backend/tracking/YOLOv8_TensorRT/pycuda_api.py)   | Enables loading and running TensorRT models with CUDA. Initializes engine and bindings from provided weights file. Supports dynamic axes and profiler setting. Conducts warm-up with predefined inputs for optimal performance.                                                       |
| [cudart_api.py](backend/tracking/YOLOv8_TensorRT/cudart_api.py)   | Enables inference acceleration using NVIDIA TensorRT for deep learning models. Initializes the engine, manages input/output bindings, supports dynamic axes, and provides a warm-up mechanism. Offers a callable interface for efficient GPU memory handling and execution.           |
| [utils.py](backend/tracking/YOLOv8_TensorRT/utils.py)             | Implements image processing functions for resizing, padding, bounding box operations, and non-maximum suppression for object detection, segmentation, and pose estimation. Enhances input data to enable efficient object localization and extraction based on confidence thresholds. |
| [common.py](backend/tracking/YOLOv8_TensorRT/common.py)           | Defines utility functions for anchor point generation and implements non-maximum suppression for object detection models. Custom module classes are provided for post-processing detection and segmentation results, along with an optimization function for model compatibility.     |
| [torch_utils.py](backend/tracking/YOLOv8_TensorRT/torch_utils.py) | Implements segmentation, pose estimation, and object detection post-processing for computer vision tasks. Performs bounding box and mask processing using Torch and torchvision ops, including non-maximum suppression.                                                               |
| [engine.py](backend/tracking/YOLOv8_TensorRT/engine.py)           | Builds a TensorRT engine from ONNX or API for object detection in image data. Handles input optimization, FP16 support, and profiling. Implements a Torch module for executing the model efficiently on GPUs, supporting dynamic shapes and profiling hooks.                          |
| [api.py](backend/tracking/YOLOv8_TensorRT/api.py)                 | Implements multiple layers like Conv2d, Bottleneck, SPPF, and Detect for the neural network in the YOLOv8_TensorRT model, enabling efficient object detection with optimized TRT operations.                                                                                          |

</details>

<details closed><summary>backend.tracking.LSTMTrack</summary>

| File                                                                             | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---                                                                              | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [LSTMTrack.py](backend/tracking/LSTMTrack/LSTMTrack.py)           | The `LSTMTrack.py` file within the `tracking` directory of the repositorys backend contains code for tracking objects using a Long Short-Term Memory (LSTM) model. This file integrates TensorFlow for deep learning and implements object tracking functionalities such as feature matching, track state management, and prediction using LSTM. Additionally, it utilizes tools for similarity measurement and optimization to enhance tracking accuracy. By leveraging these techniques, the code facilitates robust object tracking within the larger system, contributing to improved association and counting capabilities for the application. |
| [LSTM_predictor.py](backend/tracking/LSTMTrack/LSTM_predictor.py) | Predicts the next state in object motion using an LSTM model. Initiates a track from unassociated measurements and runs LSTM prediction steps for sequences. Handles bounding box coordinates and feature vectors to make accurate predictions in a 516-dimensional state space.                                                                                                                                                                                                                                                                                                                                                                                    |

</details>

<details closed><summary>backend.tracking.fast_reid</summary>

| File                                                                                       | Summary                                                                                                                                                                                                                      |
| ---                                                                                        | ---                                                                                                                                                                                                                          |
| [fast_reid_interfece.py](backend/tracking/fast_reid/fast_reid_interfece.py) | Facilitates real-time person recognition using a pre-trained model. Processes image patches, runs predictions, and handles network input adaptation. Enables feature extraction for subsequent analysis and decision-making. |

</details>



---

##  Getting Started

**System Requirements:**

* **Python**: `version 3.10.0`



###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the  repository:
>
> ```console
> $ git clone https://github.com/rolson24/BWCT-tracker.git
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd 
> ```
>
> 3. Install the dependencies:  
>> 3.a
>> ```console
>> $ pip install -r requirements.txt
>> ```  
>> 3.b Install onnx for your system. Follow these [instructions](https://pypi.org/project/onnx/).  
>>
>> 3.c Install the correct requirements for your gpu:
>>> If you have an NVIDIA gpu, you can leave the requirements file the same.  
>>>
>>> If no NVIDIA gpu, then comment out cupy, faiss-gpu, and onnxruntime-gpu from requirements.txt also change BWCT_app.py to set device to "cpu"  
>>>
>>> If you have an amd gpu, install [onnxruntime-directml](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) (First try pip install onnxruntime-directml, and if that doesn't work, then try building from [source](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).)
>
> Depends on FFMPEG: [Instructions](https://www.hostinger.com/tutorials/how-to-install-ffmpeg)
>
> 4. Install electron:
>
>> First install node.js:
>> [Instructions](https://nodejs.org/en/download)  
>>
>> Now install fs-extra:
>> ```
>> $ npm install fs-extra
>> ```
>> Now install electron:
>> ```console
>> $ npm install -g electron
>> ```  
>> 5. Start app:
>> ```console
>> $ cd path/to/BWCT-Tracker
>> $ npm start
>> ```

<h4>For Nvidia Jeton</h4>  

> Follow the software portion of these [instructions](https://docs.google.com/document/d/1U1khoDzxc9aadaoIp-lEI_-omGDv7U1ePb3fUbNVBNQ/edit?usp=sharing) to do the full setup on an Nvidia Jetson (includes instructions for OS install)

###  Usage

<h4>From <code>source</code></h4>

> Follow the build instructions above to install the project.

###  Train new YOLOv8 model
> 1. Follow this Google Colab notebook to train a new YOLOv8 model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mhtpc5kIbJU0WGojF5d9Xd8N0y4SJ5RF?usp=sharing)  
> 2. Download the trained model weights from the Google Drive folder and place them in the `backend/tracking/models` directory  
>> 2.5. (Optional) If you want the runtime to be fast and you have an NVIDIA GPU or an NVIDIA Jetson, run these commands to convert the model to a TensorRT model (first cd to the BWCT-Tracker repository and ensure you have tensorRT install):   
>> A:  
python3 backend/tracking/YOLOv8_TensorRT/export-det.py \
--weights {path/to/weights_file} \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk {num_of_classes_in_model} \
--opset 17 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0  
B:  
/usr/src/tensorrt/bin/trtexec \
--onnx={path/to/onnx_export_output} \
--saveEngine=yolov8s.engine \
--fp16  
C:  
Move the ".engine" model to the models folder  
> 3. Update the `model_path` variable in `backend/tracking/BWCT_app.py` with the path to the new model weights

---

##  Project Roadmap

- [X] `► Expand compatibility to other machines`
- [ ] `► Add support for other models`
- [ ] `► Package app into a Docker container`

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://local/BWCT-Tracker-test/issues)**: Submit bugs found or log feature requests for the `` project.
- **[Submit Pull Requests](https://local/BWCT-Tracker-test/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://local/BWCT-Tracker-test/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your local account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone ../
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to local**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="center">
   <a href="https://local{/BWCT-Tracker-test/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=BWCT-Tracker-test">
   </a>
</p>
</details>

---

##  License

This project is protected under the [AGPL-3.0](https://opensource.org/license/agpl-v3) License. For more details, refer to the [LICENSE](https://github.com/rolson24/BWCT-Tracker/tree/electron-app/LICENSE) file.

---


##  Special thanks to the following people for their existing work:
- The [Ultralytics](https://github.com/ultralytics/ultralytics) team for all their amazing work developing YOLOv8.
- Daniel Stadler and Jurgen Beyerer for their [paper](https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Stadler_An_Improved_Association_Pipeline_for_Multi-Person_Tracking_CVPRW_2023_paper.pdf) on Improved Association Tracker
- Hyeonchul Jung, Seokjun Kang, Takgen Kim, and HyeongKi Kim for their [paper](https://openaccess.thecvf.com/content/WACV2024/papers/Jung_ConfTrack_Kalman_Filter-Based_Multi-Person_Tracking_by_Utilizing_Confidence_Score_of_WACV_2024_paper.pdf) on ConfTrack. (See also their implementation [here](https://github.com/Hyonchori/ConfTrack_WACV2024))
- He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao for developing FastReID. (See also their implementation [here](https://github.com/JDAI-CV/fast-reid))
- Adrian Rosebrock, Kumar Ujjawal and Adam Spannbauer for their implementiation of fast color transfer. (See also their implementation [here](https://github.com/jrosebr1/color_transfer))
- The [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT) team for their wonderful work making YOLOv8 fast with tensorRT.
- The [Roboflow Supervision](https://roboflow.com/) team for their amazing work developing the [Supervision](https://github.com/roboflow-ai/supervision) tool.

[**Return**](#-overview)

---
