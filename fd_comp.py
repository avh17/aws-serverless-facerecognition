import os
from dotenv import load_dotenv
import json
import base64
import signal
import sys
import tempfile
from PIL import Image
from facenet_pytorch import MTCNN
import boto3
from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
from awsiot.greengrasscoreipc.model import QOS

req_queue = "https://sqs.us-east-1.amazonaws.com/221082179934/1233384011-req-queue"
resp_queue = "https://sqs.us-east-1.amazonaws.com/221082179934/1233384011-resp-queue"
mqtt_topic = "clients/1233384011-IoTThing"
op_tmp_dir = "/tmp/outputs"
load_dotenv()
access_key = os.getenv("access_key")
secret_key = os.getenv("secret_key")

class face_detection:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    def face_detection_func(self, test_image_path, output_path):
        img = Image.open(test_image_path).convert("RGB")
        key = os.path.splitext(os.path.basename(test_image_path))[0]
        face, prob = self.mtcnn(img, return_prob=True, save_path=None)

        if face is not None:
            os.makedirs(output_path, exist_ok=True)
            face_img_tensor = face - face.min()
            face_img_tensor = face_img_tensor / face_img_tensor.max()
            face_img_numpy = (face_img_tensor * 255).byte().permute(1, 2, 0).cpu().numpy()
            face_pil = Image.fromarray(face_img_numpy, mode="RGB")
            face_img_path = os.path.join(output_path, f"{key}_face.jpg")
            face_pil.save(face_img_path)
            return face_img_path
        else:
            print(f"No face is detected in {os.path.basename(test_image_path)}")
            return None

def message_handler(event, sqs_client, face_detector_instance):
    try:
        payload = json.loads(event.message.payload.decode('utf-8'))
        print(f"Received message: {payload}")

        encoded_image = payload.get("encoded")
        request_id = payload.get("request_id")
        filename = payload.get("filename")

        if not all([encoded_image, request_id, filename]):
            print("Missing required fields in message (encoded, request_id, filename). Skipping.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            input_image_path = os.path.join(temp_dir, filename)

            with open(input_image_path, "wb") as image_file:
                image_file.write(base64.b64decode(encoded_image))

            detected_face_path = face_detector_instance.face_detection_func(input_image_path, op_tmp_dir)

            if detected_face_path:
                with open(detected_face_path, "rb") as f_face:
                    encoded_face_image = base64.b64encode(f_face.read()).decode('utf-8')

                sqs_payload = {
                    "request_id": request_id,
                    "content": encoded_face_image,
                    "filename": os.path.basename(detected_face_path)
                }
                print("HELLO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                sqs_client.send_message(
                    QueueUrl=req_queue,
                    MessageBody=json.dumps(sqs_payload)
                )
                print(f"Sent detected face for request_id {request_id} to request queue.")
                os.remove(detected_face_path)
            else:
                sqs_payload = {
                    "request_id": request_id,
                    "result": "No-Face",
                    "filename": filename
                }
                sqs_client.send_message(
                    QueueUrl=resp_queue,
                    MessageBody=json.dumps(sqs_payload)
                )
                print(f"Sent 'No-Face' for request_id {request_id} to response queue.")
    except Exception as e:
        print(f"Error handling message: {str(e)}")

def handle_sigterm(*args):
    print("Gracefully shutting down...")
    sys.exit(0)

def main():
    ipc_client = GreengrassCoreIPCClientV2()

    sqs_client = boto3.client(
    "sqs", region_name="us-east-1",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key)

    face_detector_instance = face_detection()

    def on_stream_event(event):
        message_handler(event, sqs_client, face_detector_instance)

    try:
        ipc_client.subscribe_to_iot_core(
            topic_name=mqtt_topic,
            qos=QOS.AT_LEAST_ONCE,
            on_stream_event=on_stream_event
        )
        print(f"Successfully subscribed to MQTT topic: {mqtt_topic}")

        signal.signal(signal.SIGTERM, handle_sigterm)  
        signal.pause()  

    except Exception as e:
        print(f"Failed to subscribe to topic: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()