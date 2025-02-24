from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
import os
import cv2

def build_model(num_classes):
    IMG_SIZE = 224
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model




def frame_splitter(video_path, output_folder, frame_rate=None):
    """
    Splits a video into frames and saves them in the specified folder.
    
    :param video_path: Path to the input video file.
    :param output_folder: Path to the folder where frames will be saved.
    :param frame_rate: Optional, specify frame extraction rate (default is all frames).
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get original video FPS
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends
        
        # Save frame if frame_rate is specified (e.g., extract 1 frame per second)
        if frame_rate is None or (frame_count % int(fps / frame_rate) == 0):
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")


