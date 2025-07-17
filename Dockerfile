FROM python:3.8-slim
WORKDIR ${LAMBDA_TASK_ROOT}

RUN apt-get update && apt-get install -y cmake ca-certificates libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN python3 -m pip install -r requirements.txt

RUN mkdir -p /tmp/.cache
ENV TORCH_HOME=/tmp/.cache/torch
ENV XDG_CACHE_HOME=/tmp/.cache/torch

COPY fd_lambda.py ${LAMBDA_TASK_ROOT}
COPY fr_lambda.py ${LAMBDA_TASK_ROOT}
COPY resnetV1_video_weights.pt ${LAMBDA_TASK_ROOT}
COPY resnetV1.pt ${LAMBDA_TASK_ROOT}

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]

# CMD ["fd_lambda.lambda_handler"]