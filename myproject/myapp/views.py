import base64
import json
import numpy as np
from io import BytesIO
from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
import io
import pickle
from .nn import NN  # Import the NN class

# Load the pre-trained model
def load_model(dnn, file_path):
    with open(file_path, 'rb') as f:
        dnn.params = pickle.load(f)
    print(f"Model loaded from {file_path}")

# Initialize the neural network and load the model
nn = NN(sizes=[784, 128, 64, 10], epochs=10, lr=0.01)  
load_model(nn, r'C:\Users\~ideapadGAMING~\Documents\MyPrograms\HomeProject\models\Mnistnn.pkl')

def home(request):   
    return render(request, 'home.html')

def preprocess_image(image_data):
    """
    Preprocess the image to match model input requirements:
    - Convert to grayscale
    - Resize to 28x28 (or whatever your model expects)
    - Normalize pixel values
    """
    # Convert the base64 string to an image
    img = Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1])))

    # Convert to grayscale and resize to 28x28 (model input size)
    img = img.convert("L").resize((28, 28))

    # Convert to numpy array and normalize (optional based on your model)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]

    # Invert colors (if necessary)
    img_array = 1 - img_array

    # Flatten the image and reshape to match the model's input shape (784,)
    img_flatten = img_array.flatten().reshape(1, -1)

    return img_flatten

def submit_url(request):
    if request.method == 'POST':
        # Get the image data from the POST request
        data = json.loads(request.body)
        image_data = data.get('image')  # Get base64 encoded image

        # Preprocess the image
        img_flatten = preprocess_image(image_data)

        # Perform forward pass through the loaded neural network model
        prediction = nn.forward_pass(img_flatten.T)  # Transpose to match the input shape

        # Get the predicted label
        predicted_label = np.argmax(prediction)

        # Get the probabilities for all digits
        probabilities = prediction.flatten()

        # Prepare the response data
        response_data = {
            "prediction": str(predicted_label),
            "probabilities": probabilities.tolist()  # Convert to list for JSON serialization
        }

        # Return the prediction and probabilities as a JSON response
        return JsonResponse(response_data)

    return JsonResponse({"message": "Invalid request"}, status=400)
