import cv2
import os
import numpy as np
from PIL import Image

# Create a recognizer object
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the face detector (ensure this XML file is in the correct path)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    # Get the list of image files in the specified directory
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    # Initialize empty lists for face samples and IDs
    faceSamples = []
    Ids = []

    # Loop through all image paths
    for imagePath in imagePaths:
        # Check if the image file is valid
        if not os.path.isfile(imagePath):
            continue
        
        # Load the image and convert it to grayscale
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')

        # Extract the ID from the image file name (e.g., "1.1.jpg" -> 1)
        try:
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
        except ValueError:
            print(f"Skipping invalid image file: {imagePath}")
            continue

        # Detect faces in the image
        faces = detector.detectMultiScale(imageNp)
        
        # If faces are found, add them to the list
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)

    return faceSamples, Ids

# Call the function to get face samples and IDs
faces, Ids = getImagesAndLabels('Training_Images')

# Check if faces were detected
if len(faces) == 0:
    print("No faces detected in the images. Please check your training images.")
else:
    print(f"Training with {len(faces)} faces.")
    
    # Train the recognizer
    recognizer.train(faces, np.array(Ids))

    # Ensure the directory exists before saving the model
    model_dir = 'Training_Image_Label'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Absolute path to save the model
    model_path = os.path.join(os.getcwd(), model_dir, 'trainner.yml')
    print(f"Saving model to: {model_path}")
    
    # Save the trained model
    recognizer.save(model_path)
    print("Training complete! Model saved as 'trainner.yml'.")
