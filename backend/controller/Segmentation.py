import cv2
import numpy as np

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
