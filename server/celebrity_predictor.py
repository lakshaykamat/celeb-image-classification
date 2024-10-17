import joblib
import numpy as np
import cv2
import pywt
import json

# Load model and class dictionary globally when the module is imported
model = None
class_dict = None

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

# Load the model and class dictionary only once, when the module is imported
def load_model_and_dict():
    global model, class_dict
    if model is None:
        try:
            model = joblib.load('artifacts/saved_model.pkl')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    if class_dict is None:
        try:
            with open('artifacts/class_dictionary.json', 'r') as file:
                class_dict = json.load(file)
            print("Class dictionary loaded successfully")
        except Exception as e:
            print(f"Error loading class dictionary: {e}")

# Function to get cropped face image if two eyes are detected
def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Image not found at {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    
    return None

# Function to predict celebrity from image
def predict_celebrity(test_image_path):
    load_model_and_dict()  # Ensure model and dictionary are loaded
    
    if model is None:
        return {"Error": "Model could not be loaded."}
    
    img = get_cropped_image_if_2_eyes(test_image_path)
    
    if img is None:
        return {"Prediction failed": "No face with two eyes detected."}
    
    scalled_raw_img = cv2.resize(img, (32, 32))
    img_har = w2d(img, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    
    combined_img = np.vstack((
        scalled_raw_img.reshape(32*32*3, 1),
        scalled_img_har.reshape(32*32, 1)
    ))
    
    X = combined_img.reshape(1, -1).astype(float)
    predicted_probabilities = model.predict_proba(X)[0]
    names = list(class_dict.keys())
    response = {}
    for i in range(len(names)):
        response[names[i]] = round(float(predicted_probabilities[i] * 100), 2)
    
    return response
