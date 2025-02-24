from flask import Flask,request,jsonify,send_file
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import numpy as np
import os
import shutil
from tqdm import tqdm
from controller.Classification import *

app=Flask(__name__)

frames_path=r"C:\BreastCancer\ChanCode\backend\tempDB\frames"
output_frames=r"C:\BreastCancer\ChanCode\backend\tempDB\output_frames"
input_video_path=r"C:\BreastCancer\ChanCode\backend\tempDB\videos"

@app.route("/test",methods=["GET"])
def test():
    return jsonify({"message":"Working"})

@app.route("/classification",methods=["POST"])
def classify():
    files=request.files
    image_data=files["image"]
    image = Image.open(image_data).resize((224, 224))
    path = r"C:\BreastCancer\ChanCode\backend\models\Efficient_Net_Final.weights.h5"
    NUM_CLASSES = 3
    model = build_model(num_classes=NUM_CLASSES)
    model.load_weights(path)
    x = np.expand_dims(image, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print(preds)
    class_labels = ['benign', 'malignant', 'normal']
    max_index = np.argmax(preds[0])
    predicted_class = class_labels[max_index]
    return jsonify({"prediction ":predicted_class})


@app.route("/generate_video",methods=["POST"])
def generator():
    video = request.files["video"]
    video_path=os.path.join(input_video_path,video.filename)
    video.save(video_path)
    frame_splitter(video_path, frames_path)
    frame_files = os.listdir(frames_path)
    for x in tqdm(frame_files, desc="Processing Frames", unit="frame"):
        frame = x
        classification = classify(frame)
        
        if classification in ["benign", "malignant"]:
            segment(frame)
        else:
            source = os.path.join(frames_path, x)
            destination = os.path.join(output_frames, x)
            shutil.move(source, destination)
    

    

# @app.route("/segmentation",methods=["POST"])
# def seg():
#     files=request.files["image"]


if(__name__=="__main__"):
    app.run(host="0.0.0.0",port=4000,debug=True)