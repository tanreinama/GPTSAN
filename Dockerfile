FROM tensorflow/tensorflow:2.5.0-gpu-jupyter
RUN git clone https://github.com/tanreinama/GPTSAN
WORKDIR /tf/GPTSAN
RUN pip install -r requirements.txt
