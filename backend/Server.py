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
from controller.Segmentation import *
from io import BytesIO

app=Flask(__name__)

UPLOAD_FOLDER = r"C:\BreastCancer\ChanCode\backend\tempDB\temp"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    for x in frame_files:
        frame = x
        path=os.path.join(frames_path,frame)
        classification = classif(path)
        
        if classification in ["benign", "malignant"]:
            continue
        else:
            source = os.path.join(frames_path, x)
            destination = os.path.join(output_frames, x)
            shutil.move(source, destination)
    output_video_path=r"C:\BreastCancer\ChanCode\backend\tempDB\videos\output\video.mpv4"
    frames_to_video(output_frames,output_video_path,video_path)
    return send_file(output_video_path, mimetype="video/mp4")

    

@app.route("/segmentation_mask",methods=["POST"])
def seg():
    files=request.files
    image_data=files["image"]
    image = Image.open(image_data).resize((224, 224))
    image_np = np.array(image)
    model_path=r"C:\BreastCancer\ChanCode\backend\models\model.keras"
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss,"f1sc":f1sc}):
        model = tf.keras.models.load_model(model_path)
    x = image_np/255.0 
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    pred = y_pred * 255
    pred_image = Image.fromarray(pred.astype(np.uint8))
    
    # Save to in-memory file
    img_io = BytesIO()
    pred_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    # Return the image
    return send_file(img_io, mimetype='image/png')
    
@app.route("/segmentation", methods=["POST"])
def seg_over():
    image_data = request.files["image"]
    image = Image.open(image_data).resize((224, 224))
    image_np = np.array(image)
    
    model_path = r"C:\BreastCancer\ChanCode\backend\models\model.keras"
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss, "f1sc": f1sc}):
        model = tf.keras.models.load_model(model_path)
    
    x = image_np/255.0 
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)
    
    # Create a single-channel mask (no need to expand and concatenate)
    mask = y_pred * 255
    
    # Create the colored mask correctly
    mask_colored = np.zeros_like(image_np)
    mask_colored[:, :, 2] = mask  # Apply mask to Red channel
    
    # Create the overlay
    overlay = cv2.addWeighted(image_np, 0.7, mask_colored, 0.3, 0)
    
    # Convert to PIL Image and return
    pred_image = Image.fromarray(overlay.astype(np.uint8))
    
    # Save to in-memory file
    img_io = BytesIO()
    pred_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    # Return the image
    return send_file(img_io, mimetype='image/png')

if(__name__=="__main__"):
    app.run(host="0.0.0.0",port=4000,debug=True)