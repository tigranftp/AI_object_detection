FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN pip install ultralytics
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN yes | pip uninstall torch torchvision
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118