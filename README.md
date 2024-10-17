# Celebrity Image Classification

A web application built with Flask that uses machine learning to classify images of celebrities. Currently supports the following celebrities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [License](#license)
- [Contributing](#contributing)

## Features

- Upload an image of a celebrity.
- Classify the image and return the prediction with confidence percentages.
- User-friendly interface using Tailwind CSS.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework in Python.
- **OpenCV**: Used for image processing and face detection.
- **Joblib**: For loading pre-trained machine learning models.
- **PyWavelets**: For applying wavelet transforms to images.
- **Tailwind CSS**: For styling the frontend.
- **HTML/CSS/JavaScript**: Basic web technologies for frontend development.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/celebrity-image-classification.git
   cd celebrity-image-classification
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or create the pre-trained model and class dictionary**:
   - Ensure that you have your `saved_model.pkl` and `class_dictionary.json` files in the `artifacts` directory.
   - Create a `test_images` directory for uploaded images.

## Usage

1. **Run the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to `http://127.0.0.1:5000/`.

3. **Upload an image** of a celebrity and click on "Classify Image" to get the predictions.

## API Endpoints

### Home Route

- **GET /**: Renders the home page with the image upload form.

### Prediction Route

- **POST /api/predict_celebrity**: Accepts an image upload and returns a JSON response with the classification results.

**Example Request**:
```bash
curl -X POST -F "image=@path_to_your_image.jpg" http://127.0.0.1:5000/api/predict_celebrity
```

**Example Response**:
```json
{
    "Lionel Messi": 85.5,
    "Maria Sharapova": 0.2,
    "Roger Federer": 5.1,
    "Serena Williams": 0.3,
    "Virat Kohli": 9.0
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.