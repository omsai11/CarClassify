from flask import Flask, render_template, request, url_for, redirect
import numpy as np
from keras.utils import load_img, img_to_array
from model1 import predictfunc
from PIL import Image



app = Flask(__name__)

global image
@app.route('/')
def index():       
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    file = request.files['file']
    if file.filename == '':
        return 'Empty filename', 400
    img=Image.open(file)
    
    img=img.resize((300,300))
    x = img_to_array(img)
    x /= 255
    x = np.expand_dims(x,axis=0)
    predictions=predictfunc(x)
    predicted_class=np.argmax(predictions[0])
    a=predicted_class
    print(a)
    return render_template('car.html',result=a)
    


if __name__=="__main__":
    app.run(debug=True,port=8000)