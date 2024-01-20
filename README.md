# Image Processing and Classification Flask App

This is a Flask web application for image processing and classification. It provides various image processing functionalities and includes a trained model for image classification.

## Prerequisites

Make sure you have the following installed:

- Python (>=3.6)
- TensorFlow
- Flask
- OpenCV
- scikit-image
- PIL (Pillow)
- NumPy

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Geting Started
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```
2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a file named .env in the project root and add the following environment variables:
```bash
SERVERTESTDIR=/path/to/server/test/directory
MODELDIR=/path/to/model/directory
LABELSDIR=/path/to/labels/directory
```
Replace /path/to/server/test/directory, /path/to/model/directory, and /path/to/labels/directory with the actual paths on your system.


## Usage
1. Run the Flask application:
```bash
python cardsClassification.py
```
2. Access the application in your browser at http://127.0.0.1:5000/.


## Endpoints
- `/CompareImage/<imgName>`: Compares the given image and returns the classification result.
- `/TrainModel/<imgName>`: Performs data augmentation on the provided image and retrains the classification model.


## Acknowledgements
- This application uses Flask for web development.
- Image classification is performed using a pre-trained InceptionV3 model.
