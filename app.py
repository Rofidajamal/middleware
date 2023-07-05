from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2


app = Flask(__name__)


heightImg = 640
widthImg = 480

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def procssing(img):
    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)  # ADD GAUSSIAN BLUR
    #thres = uti.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur,0,0)  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    ## FIND ALL COUNTOURS
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    biggest, maxArea = biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    biggest = reorder(biggest)
    x=biggest[0][0][0]
    w=biggest[1][0][0]
    y=biggest[0][0][1]
    h=biggest[3][0][1]
    #print(x,y,w,h)
    imgBigContour=imgBigContour[y:h,x:w]
    imgBigContour = cv2.cvtColor(imgBigContour, cv2.COLOR_BGR2GRAY)
    imgBigContour = cv2.resize(imgBigContour, (224, 224))
    imgBigContour = imgBigContour.reshape(imgBigContour.shape[0], imgBigContour.shape[1], 1)

    return  imgBigContour


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def preprocessing(image):
    img=Image.open(image)
    img=np.array(img)
    img = Image.fromarray(img)
    img.save('image_gray.png')
    img=cv2.imread('image_gray.png')
    img=procssing(img)
    img=adjust_gamma(img,1.3)
    img.shape=(1,224,224,1)
    img_arr=np.array(img)
    return img_arr

classes = [
    'COVID-19 Patient',
    ' Myocardial Infarction Patient',
    ' Patient that have History of MI',
    'Patient that have abnormal heart beats',
    'Normal Person'
]

model = tf.keras.models.load_model("efficientnetv3with5foldAPI.h5")

@app.route('/')
def index():
    return render_template('img.html')

@app.route('/predictApi', methods=["POST"])
def API():
    try:
        if 'fileup' not in request.files:
            return "Please Try again. The image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprocessing(image)
        result = model.predict(image_arr)
        ind = np.argmax(result)
        prediction = classes[ind]
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'error': 'Try again'})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        image = request.files['fileup']
        image_arr = preprocessing(image)
        result = model.predict(image_arr)
        print (result)
        ind = np.argmax(result)
        print(ind)
        prediction = classes[ind]
        print(prediction)
        return render_template('img.html', prediction=prediction)
    else:
        return render_template('img.html')

if __name__ == '__main__':
    app.run(debug=True)
