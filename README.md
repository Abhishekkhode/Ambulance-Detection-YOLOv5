# Ambulance Detection System with YOLOv5

This project demonstrates a real-time object detection system trained to identify ambulances in video and image feeds using the YOLOv5 deep learning model.

## Features

- Real-time ambulance detection using a webcam.
- High-performance inference on GPU.
- Trained using a custom dataset of various ambulance types and environmental conditions.
- Utilizes transfer learning for efficient model training.

## Getting Started

Follow these steps to set up the project environment on your local machine.

### Prerequisites

* **Git:** For cloning the repositories.
* **Conda:** For creating an isolated Python environment.
* **A CUDA-enabled NVIDIA GPU** (e.g., RTX 3050) for faster training and detection.

### Installation

1.  **Clone the YOLOv5 repository** and your project.
    ```bash
    git clone [https://github.com/ultralytics/yolov5.git](https://github.com/ultralytics/yolov5.git)
    ```

2.  **Create and activate a new Conda environment** for the project.
    ```bash
    conda create -n yolov5_env python=3.9
    conda activate yolov5_env
    ```

3.  **Install PyTorch with CUDA support.** This is a critical step for GPU acceleration.
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
4.  **Install the remaining requirements.**
    ```bash
    pip install -r yolov5/requirements.txt
    ```

## Usage

To use the trained model for detection, you must first have the `best.pt` file. This file should be placed in the `yolov5/runs/train/yambulance_detection_fine-tune2/weights/` directory.

### Live Webcam Detection

To run live detection using your webcam, navigate to the `yolov5` directory and use the `detect.py` script.

```bash
python detect.py --weights runs/train/yambulance_detection_fine-tune2/weights/best.pt --source 0 --conf-thres 0.5



# For a single image
python detect.py --weights runs/train/yambulance_detection_fine-tune2/weights/best.pt --source your_image_file.jpg --conf-thres 0.5

# For a video
python detect.py --weights runs/train/yambulance_detection_fine-tune2/weights/best.pt --source your_video_file.mp4 --conf-thres 0.5



To Train
python yolov5/train.py --img 640 --batch 4 --epochs 100 --data your_dataset/data.yaml --weights yolov5s.pt --cfg yolov5s.yaml --name ambulance_training





