# Face Recognition Service on AWS Lambda

## Overview

This project implements a serverless face recognition service using AWS Lambda, SQS, and ECR. The application provides face recognition as a service on video frames streamed from clients (e.g., security cameras) using a multi-stage pipeline.

## Architecture

<img width="1788" height="596" alt="image" src="https://github.com/user-attachments/assets/255653f2-2419-4a8f-b1ef-4097f495c1aa" />


The system consists of two main Lambda functions connected via SQS queues:

1. **Face Detection Function** (`face-detection`)
   - Receives video frames from clients via HTTP POST
   - Performs face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
   - Sends detected faces to SQS request queue

2. **Face Recognition Function** (`face-recognition`)
   - Triggered by messages in the SQS request queue
   - Performs face recognition using ResNet with VGGFace2 pretrained model
   - Sends results to SQS response queue

## Prerequisites

- AWS Account with appropriate permissions
- Docker installed for building container images
- Python 3.x
- AWS CLI configured

## Setup Instructions

### Step 1: Set up ECR (Elastic Container Registry)

1. Create an ECR repository for your Lambda functions
2. Use the provided Dockerfile template with the following required packages:
   - `boto3`
   - `facenet_pytorch`
   - `awslambdaric`
   - `requests`
   - `Pillow`
   - `opencv-python`

3. Build and push your Docker image to ECR:
   ```bash
   docker build -t your-ecr-repo .
   docker tag your-ecr-repo:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/your-ecr-repo:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/your-ecr-repo:latest
   ```

### Step 2: Set up SQS Queues

Create two SQS queues in the US-East-1 region:

1. **Request Queue**: `-req-queue`
   - Used by face-detection Lambda to send detected faces
   - Triggers the face-recognition function

2. **Response Queue**: `-resp-queue`
   - Used by face-recognition Lambda to send results
   - Polled by clients for classification results

### Step 3: Create Face Detection Lambda Function

**Configuration:**
- Function name: `face-detection`
- Runtime: Use your ECR container image
- Create a Function URL for HTTP access
- Required IAM permissions:
  - `AWSLambdaSQSQueueExecutionRole`
  - `AWSLambdaVPCAccessExecutionRole`

**Functionality:**
- Accepts POST requests with JSON body containing:
  - `content`: Base64-encoded input image
  - `request_id`: Unique identifier for the request
  - `filename`: Name of the input image
- Uses MTCNN for face detection
- Sends detected faces to SQS request queue

### Step 4: Create Face Recognition Lambda Function

**Configuration:**
- Function name: `face-recognition`
- Runtime: Use your ECR container image
- SQS trigger: Connected to the request queue
- Required IAM permissions:
  - `AWSLambdaSQSQueueExecutionRole`
  - `AWSLambdaVPCAccessExecutionRole`

**Functionality:**
- Triggered by SQS messages from face-detection function
- Uses ResNet model: `InceptionResnetV1(pretrained='vggface2').eval()`
- Computes face embeddings and matches against known faces
- Sends results to response queue with format:
  ```json
  {
    "request_id": "<request_id>",
    "result": "<classification_result>"
  }
  ```

## Implementation Details

### Face Detection (fd_lambda.py)

The face detection function:
1. Extracts base64-encoded image from the request body
2. Decodes and processes the image
3. Applies MTCNN for face detection
4. Extracts detected face regions
5. Sends face data to SQS request queue

### Face Recognition (fr_lambda.py)

The face recognition function:
1. Receives face data from SQS request queue
2. Initializes ResNet model with VGGFace2 pretrained weights
3. Computes face embeddings
4. Compares with known face embeddings
5. Returns the closest match
6. Sends results to SQS response queue

## File Structure

```
face-recognition-service/
├── face-detection/
│   └── fd_lambda.py
├── face-recognition/
│   └── fr_lambda.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Configuration

**Environment Variables:**
- `REQUEST_QUEUE_URL`: SQS request queue URL
- `RESPONSE_QUEUE_URL`: SQS response queue URL
- `AWS_REGION`: AWS region (us-east-1)

**Required Dependencies:**
```txt
boto3
facenet_pytorch
awslambdaric
requests
Pillow
opencv-python
torch
torchvision
```

## Development Best Practices

1. Use the same Docker image for both Lambda functions (override CMD)
2. Use `python:slim` base image to reduce size
3. Install CPU-only versions of PyTorch/Torchvision
4. Clean up SQS queues during development to avoid recursive triggers
5. Use US-East-1 region for all resources

## Security

- Ensure proper IAM roles and policies
- Use least privilege principle
- Secure your credentials and access keys

## Troubleshooting

1. **SQS Messages Stuck**: Purge queues or disable triggers during development
2. **Lambda Timeouts**: Optimize model loading and processing
3. **Image Size Issues**: Use smaller base images and only required packages
4. **Permission Errors**: Verify IAM roles and policies are correctly attached
5. **Model Loading Issues**: Ensure PyTorch and facenet_pytorch are properly installed

## API Usage

### Face Detection Endpoint

**POST** to Lambda Function URL

**Request Body:**
```json
{
  "content": "base64_encoded_image_data",
  "request_id": "unique_request_identifier",
  "filename": "image_filename.jpg"
}
```

**Response:**
```json
{
  "statusCode": 200,
  "body": "Face detection completed"
}
```

### Response Queue Format

**SQS Response Message:**
```json
{
  "request_id": "unique_request_identifier",
  "result": "recognized_person_name"
}
```
