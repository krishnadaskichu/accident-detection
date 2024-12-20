# pip install ultralytics
# pip install tqdm --upgrade
import os
import shutil
import random
from tqdm.notebook import tqdm
from ultralytics import YOLO
# !pwd
train_img_path = "./yolo_data/images/train/"
train_label_path = "./yolo_data/labels/train/"
val_img_path = "./yolo_data/images/val/"
val_label_path = "./yolo_data/labels/val/"
def train(path):
  files=list(set([name[:-4] for name in os.listdir(path)]))

  #creating directories

  os.makedirs(train_img_path,exist_ok=True)
  os.makedirs(train_label_path,exist_ok=True)




#copy train images and labels to correspondig folders

  for file in tqdm(files):
    shutil.copy2(path+file+'.jpg',train_img_path+file+'.jpg')
    shutil.copy2(path+file+'.txt',train_label_path+file+'.txt')



def val(path):
  files=list(set([name[:-4] for name in os.listdir(path)]))

  #creating directories

  os.makedirs(val_img_path,exist_ok=True)
  os.makedirs(val_label_path,exist_ok=True)

#copy train images and labels to correspondig folders

  for file in tqdm(files):
    shutil.copy2(path+file+'.jpg',val_img_path+file+'.jpg')
    shutil.copy2(path+file+'.txt',val_label_path+file+'.txt')




train("/content/drive/MyDrive/accident_detection/accident_detection_dataset/images_train/")
val("/content/drive/MyDrive/accident_detection/accident_detection_dataset/images_valid/")



!yolo task=detect mode=train model=yolov8n.pt data=/content/drive/MyDrive/accident_detection/accident_detection_dataset/dataset.yaml  epochs=10

!yolo task=detect mode=val model=yolov8n.pt source="/content/drive/MyDrive/accident_detection/accident_detection_dataset/Training_result/accident_predicton2/weights/best.pt" data=/content/drive/MyDrive/accident_detection/accident_detection_dataset/dataset.yaml

!yolo task=detect mode=predict model=/content/drive/MyDrive/accident_detection/accident_detection_dataset/Training_result/accident_predicton2/weights/best.pt conf=0.25 source=/content/drive/MyDrive/accident_detection/accident_detection_dataset/images_test


import subprocess

def run_yolo_command():
    command = [
        "yolo",
        "task=detect",
        "mode=predict",
        "model=/content/drive/MyDrive/accident_detection/accident_detection_dataset/Training_result/accident_predicton2/weights/best.pt",
        "conf=0.25",
        "source=/content/drive/MyDrive/accident_detection/accident_detection_dataset/images_test"
    ]

    try:
        # Run the command
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())  # Print the command output
        return True
    except subprocess.CalledProcessError as e:
        # Handle error
        print(f"Command failed with error: {e.stderr.decode()}")
        return False

# Store the result in a boolean variable
yolo_success = run_yolo_command()

print(f"YOLO command successful: {yolo_success}")



# !pip install twilio
# !whatsapp-python code send.py

model = YOLO('yolov8n-seg.pt')
model.predict(source='/content/drive/MyDrive/accident_detection/accident_detection_dataset/images_test/4026_jpeg.rf.6475d9b651132eb567a50bbe26b7b706.jpg')


!cp -r /content/runs/detect/predict /content/drive/MyDrive/accident_detection/accident_detection_dataset/output


# !pip install twilio

from twilio.rest import Client
if(yolo_success):
  account_sid = 'AC80a7eb2fb93c13155ed687ce9c39bb97'
  auth_token = 'a26dbf30824b1cada250b145532f12bd'
  client = Client(account_sid, auth_token)

  message = client.messages.create(
  from_='whatsapp:+14155238886',
  body='Severe accident ',
  media_url='/content/drive/MyDrive/accident_detection/accident_detection_dataset/images_test/4026_jpeg.rf.6475d9b651132eb567a50bbe26b7b706.jpg',
  to='whatsapp:+918281884477'
)

  print(message.sid)
else:
  print("not severe")

