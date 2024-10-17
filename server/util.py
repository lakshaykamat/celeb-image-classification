import joblib
import numpy as np
import cv2
import pywt
import json
import os

# Function for Wavelet Transform (w2d)
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

# Ensure correct paths to haarcascade files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load the class dictionary from JSON
with open('artifacts/class_dictionary.json', 'r') as file:
    class_dict = json.load(file)

# Load the pre-trained model
try:
    model = joblib.load('artifacts/saved_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to get cropped face image if two eyes are detected
def get_cropped_image_if_2_eyes(image_path):
    # Read the image from the file
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Image not found at {image_path}")
        return None
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Loop through detected faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # If two or more eyes are detected, return the cropped face
        if len(eyes) >= 2:
            return roi_color  # Return the cropped face image
    
    # If no face with two eyes is found, return None
    return None

# Function to predict celebrity from image
def predict_celebrity(test_image_path, model):
    if model is None:
        return {"Error": "Model could not be loaded."}
    
    # Get cropped image
    img = get_cropped_image_if_2_eyes(test_image_path)
    
    if img is None:
        return {"Prediction failed": "No face with two eyes detected."}
    
    # Resize the cropped image to (32, 32)
    scalled_raw_img = cv2.resize(img, (32, 32))

    # Apply wavelet transform
    img_har = w2d(img, 'db1', 5)  # Ensure the w2d function is defined

    # Resize the wavelet-transformed image to (32, 32)
    scalled_img_har = cv2.resize(img_har, (32, 32))

    # Combine raw and wavelet-transformed image
    combined_img = np.vstack((
        scalled_raw_img.reshape(32*32*3, 1),
        scalled_img_har.reshape(32*32, 1)
    ))

    # Prepare the image for prediction
    X = combined_img.reshape(1, -1).astype(float)
    
    # Predict the class probabilities
    predicted_probabilities = model.predict_proba(X)[0]  # Get probabilities for all classes
    names = list(class_dict.keys())
    response = {}
    for i in range(len(names)):
        response[names[i]] = round(float(predicted_probabilities[i] * 100), 2)
    
    return response

# Test the prediction
test_image_path = 'test_images/virat1.webp'
if os.path.exists(test_image_path):
    prediction_percentages = predict_celebrity(test_image_path, model)
    print(prediction_percentages)
else:
    print(f"Test image not found at {test_image_path}")
