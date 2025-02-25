import cv2
import numpy as np
import tensorflow as tf
H = 224
W = 224

def save_results(image, mask, y_pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    line = np.ones((H, 10, 3)) * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


def overlay(input_image,predicted_mask,save_path):
    original_image = cv2.imread(input_image, cv2.IMREAD_COLOR)
    original_image = cv2.resize(original_image, (224, 224))

    
    mask = cv2.imread(predicted_mask, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (224, 224))

    # Convert mask to color (Red for visibility)
    mask_colored = np.zeros_like(original_image)
    mask_colored[:, :, 2] = mask  # Apply mask to Red channel

    # Blend the original image with the mask (Overlay effect)
    overlay = cv2.addWeighted(original_image, 0.7, mask_colored, 0.3, 0)

    cv2.imwrite(save_path+"output.png", overlay)

def segment(frame_path):
    return "hi"

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def f1sc(y_true, y_pred):
    # Ensure tensors have the same data type
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Flatten tensors if necessary (for compatibility with some metrics)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # Calculate precision and recall
    precision = tf.math.divide_no_nan(tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)), tf.reduce_sum(y_pred))
    recall = tf.math.divide_no_nan(tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)), tf.reduce_sum(y_true))

      # Calculate F1-score (avoid division by zero)
    f1 = 2 * tf.math.divide_no_nan(precision * recall, precision + recall + 1e-10)

    return f1