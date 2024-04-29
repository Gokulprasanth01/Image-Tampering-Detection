from flask import Flask, render_template, request
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from helper import prepare_image_for_ela, prerpare_img_for_weather
import numpy as np
from fetchOriginal import image_coordinates, get_weather

app = Flask(__name__)

class_weather = ['Lightning', 'Rainy', 'Snow', 'Sunny']
class_ELA = ['Real', 'Tampered']

from flask import Flask, render_template, request
import cv2
import os
import numpy as np

app = Flask(__name__)

def restore_image(tampered_image_path, mask):
    tampered_image = cv2.imread(tampered_image_path)
    
    # Apply image inpainting using the mask
    restored_image = cv2.inpaint(tampered_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return restored_image

@app.route('/restore_image', methods=['POST'])
def restore_image_route():
    # Get tampered image from form submission
    tampered_img_file = request.files['tampered_img_file']
    tampered_img_name = 'static/temp_tampered_image.jpg'  # Temporary tampered image name
    tampered_img_file.save(tampered_img_name)  # Save uploaded tampered image

    # Load the tampered imag
    tampered_image = cv2.imread(tampered_img_name)

    # Create a mask based on the tampered image
    mask = np.zeros_like(tampered_image[:, :, 0])

    # Assuming the blurred region is defined by a bounding box (x, y, w, h)
    # You may need a more sophisticated method to identify this region in practice
    x, y, w, h = 100, 100, 200, 200  # Example bounding box coordinates

    # Draw a white rectangle on the mask to represent the region to be restored
    cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)

    # Apply restoration using the mask
    restored_image = restore_image(tampered_img_name, mask)

    # Save the restored image
    restored_image_path = 'static/restored_image.jpg'
    cv2.imwrite(restored_image_path, restored_image)

    return render_template('restored_image.html', restored_img=restored_image_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get image from form submission
        img_file = request.files['img_file']
        img_name = 'static/temp_image.jpg'  # Temporary image name
        img_file.save(img_name)  # Save uploaded image
        
        # Process the uploaded image
        org = cv2.imread(img_name)
        org_resized = cv2.resize(org, (750, 750))  # Resize org to match the size of blank
        res1 = detect_ELA(img_name)
        res2 = ''
        res3 = ''

        flag = request.form.get('outdoor_flag', '')
        if flag == 'y':
            res2 = org_weather(img_name)
            res3 = detect_weather(img_name)

        # Creating blank image for displaying results
        blank = np.zeros((750, 750, 3), np.uint8)
        cv2.putText(blank, res1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
        cv2.putText(blank, res2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
        cv2.putText(blank, res3, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
        final_res = np.concatenate((org_resized, blank), axis=1)
        cv2.imwrite('static/result.jpg', final_res)

        return render_template('result.html', tampered_img=img_name)

    return render_template('index.html')


def detect_ELA(img_name):
    res1 = ''
    # Preparing img for ELA_model
    np_img_input = prepare_image_for_ela(img_name)

    # Load model
    ELA_model = load_model('ELA_Training/Model.h5')

    # Test image
    # Reshape input to match model's expected shape
    np_img_input = np.expand_dims(np_img_input, axis=0)

    Y_predicted = ELA_model.predict(np_img_input, verbose=0)

    res1 += "1. Model shows {}% accuracy of image being {}".format(round(np.max(Y_predicted[0]) * 100), class_ELA[np.argmax(Y_predicted[0])])
    return res1

def detect_weather(img_name):
    res3 = ''
    np_img_input = prerpare_img_for_weather(img_name)

    model_Weather = load_model('WeatherCNNTraining/Weather_Model.h5')

    # Test image
    # Reshape input to match model's expected shape
    np_img_input = np.expand_dims(np_img_input, axis=0)
    np_img_input = np.squeeze(np_img_input, axis=0)  # Remove extra dimension

    Y_predicted = model_Weather.predict(np_img_input, verbose=0)

    res3 += "3. Model shows weather in Image is {}".format(class_weather[np.argmax(Y_predicted[0])])
    return res3

def org_weather(img_name):
    res2 = ''
    date_time, lat, long = image_coordinates(img_name)
    res2 += "2. Weather originally was {}".format(get_weather(date_time, lat, long))
    return res2

if __name__ == '__main__':
    app.run(debug=True)
