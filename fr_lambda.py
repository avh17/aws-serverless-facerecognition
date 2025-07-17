import os
import time
import json
import boto3
import torch
import base64
import tempfile
import shutil
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont

class face_recognition:
    def __init__(self):
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def face_recognition_func(self, model_path, model_wt_path, face_img_path):

        # Step 1: Load image as PIL
        face_pil = Image.open(face_img_path).convert("RGB")
        key      = os.path.splitext(os.path.basename(face_img_path))[0].split(".")[0]

        # Step 2: Convert PIL to NumPy array (H, W, C) in range [0, 255]
        face_numpy = np.array(face_pil, dtype=np.float32)  # Convert to float for scaling

        # Step 3: Normalize values to [0,1] and transpose to (C, H, W)
        face_numpy /= 255.0  # Normalize to range [0,1]

        # Convert (H, W, C) → (C, H, W)
        face_numpy = np.transpose(face_numpy, (2, 0, 1))

        # Step 4: Convert NumPy to PyTorch tensor
        face_tensor = torch.tensor(face_numpy, dtype=torch.float32)

        saved_data = torch.load(model_wt_path)  # loading resnetV1_video_weights.pt

        self.resnet = torch.jit.load(model_path) # this uses the model trace. resnetV1.pt

        if face_tensor != None:
            emb             = self.resnet(face_tensor.unsqueeze(0)).detach()  # detech is to make required gradient false
            embedding_list  = saved_data[0]  # getting embedding data
            name_list       = saved_data[1]  # getting list of names
            dist_list       = []  # list of matched distances, minimum distance is used to identify the person

            for idx, emb_db in enumerate(embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)

            idx_min = dist_list.index(min(dist_list))
            return name_list[idx_min]
        else:
            print(f"No face is detected")
            return

def lambda_handler(event, context):
    try:
        recognizer = face_recognition()
        
        for record in event.get("Records", []):
            body = json.loads(record["body"])
            request_id = body.get("request_id")
            content = body.get("content")
            filename = body.get("filename")

            extracted_image_data = base64.b64decode(content)
            temp_dir = tempfile.mkdtemp()
            try:
                image_path = os.path.join(temp_dir, filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                with open(image_path, "wb") as f:
                    f.write(extracted_image_data)

                result = recognizer.face_recognition_func("resnetV1.pt", "resnetV1_video_weights.pt", image_path)
            finally:
                shutil.rmtree(temp_dir)

            if result:
                response = {
                    "request_id": request_id,
                    "result": result,
                }
            else:
                return {
                    "statusCode": 200,
                    "body": json.dumps({"message": "No face is detected"})
                }
    
            queue_url = "https://sqs.us-east-1.amazonaws.com/221082179934/1233384011-resp-queue"
            sqs = boto3.client("sqs", region_name="us-east-1")
        
            sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(response)
            )
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Processed requests successfully."})
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }