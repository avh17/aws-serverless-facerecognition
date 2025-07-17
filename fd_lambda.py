import os
import json
import base64
import boto3
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw, ImageFont
import tempfile
import shutil
import io

class face_detection:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    def face_detection_func(self, test_image_path, output_path):

        img     = Image.open(test_image_path).convert("RGB")
        img     = np.array(img)
        img     = Image.fromarray(img)

        key = os.path.splitext(os.path.basename(test_image_path))[0].split(".")[0]

        face, prob = self.mtcnn(img, return_prob=True, save_path=None)

        if face != None:

            os.makedirs(output_path, exist_ok=True)

            face_img = face - face.min()  
            face_img = face_img / face_img.max()  
            face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()  


            face_pil        = Image.fromarray(face_img, mode="RGB")
            face_img_path   = os.path.join(output_path, f"{key}_face.jpg")

            face_pil.save(face_img_path)
            return face_img_path        
        else:
            print(f"No face is detected")
            return

def lambda_handler(event, context):
    try:
        print("Received event", event)
        detector = face_detection()

        body = json.loads(event['body'])
        content = body.get("content")
        request_id = body.get("request_id")
        file_name = body.get("filename")
        filename = os.path.basename(file_name)
        output_path = "/tmp/outputs"

        print(f"Request {request_id}: Processing file {filename}")
        temp_dir = tempfile.mkdtemp()
        try: 
            img_path = os.path.join(temp_dir, filename)

            with open(img_path, "wb") as image_file:
                image_file.write(base64.b64decode(content))

            face_image_path = detector.face_detection_func(img_path, output_path)

            if face_image_path:
                with open(face_image_path, "rb") as f:
                    encoded_face = base64.b64encode(f.read()).decode('utf-8')
        finally:
            shutil.rmtree(temp_dir)
    
        if encoded_face:
            response = {
                "request_id": request_id,
                "content": encoded_face,
                "filename": filename
            }
        else:
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No face detected"})
            }
    
        queue_url = "https://sqs.us-east-1.amazonaws.com/221082179934/1233384011-req-queue"
        sqs = boto3.client("sqs", region_name="us-east-1")
    
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(response)
        )

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Processed and queued successfully"})
        }
    except Exception as e:
        print("Error:", e)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": "Error processing request",
                "error": str(e)
            })
        }

