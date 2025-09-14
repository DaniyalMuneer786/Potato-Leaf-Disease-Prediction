from flask import Flask, render_template, request
import joblib
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

# Load your trained model
loaded_model = joblib.load('my_model.pkl')


app = Flask(__name__)

@app.route('/')
def form():
    return render_template('Form.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        if 'image' not in request.files:
            return "No image uploaded"
        
        image = request.files['image']
        
        if image.filename == '':
            return "Empty file name"
        
        filename = secure_filename(image.filename)
        image_path = f'static/{filename}'
        image.save(image_path)

        # Preprocess the image for prediction (you may need to adjust this based on your model requirements)
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize the image to match your model's input size
        img_array = np.asarray(img)  # Convert image to numpy array
        img_array = img_array / 255.0  # Normalize the image array if required
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Define a dictionary to map predicted class indices to labels
        class_mapping = {0: 'Early blight', 1: 'Healthy', 2: 'Late blight'}

        # Perform prediction using your model
        prediction = loaded_model.predict(img_array)
        predicted_class_index = np.argmax(prediction)  # Get the index of the class with the highest probability
        predicted_class = class_mapping[predicted_class_index]  # Map predicted class index to label
        

        # Render Result.html with the image and its predicted label
        return render_template('Result.html', image=image, predicted_class=predicted_class)
    else:
        return "GET request not supported"

if __name__ == '__main__':
    app.run(debug=True)
