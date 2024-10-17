from flask import Flask, render_template, jsonify, request
from celebrity_predictor import predict_celebrity
import os

# Create the Flask app
app = Flask(__name__)

# Create a folder for uploaded images
UPLOAD_FOLDER = 'test_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict_celebrity', methods=['POST'])  # Change to POST
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Call the prediction function
    response = predict_celebrity(image_path)
    return jsonify(response)

if __name__ == '__main__':
    # Run the Flask app on localhost port 5000
    app.run(debug=True)
