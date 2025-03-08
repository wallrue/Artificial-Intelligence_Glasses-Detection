# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:03:37 2022

@author: mylocalaccount
"""

import cv2
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model=load_model("my_model.keras")

size = 100
font = cv2.FONT_HERSHEY_SIMPLEX
label=['no_glasses','glasses'] #POS is 1, NEG is 0

vid = cv2.VideoCapture(0)

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


while True:
    # Read the frame
    _, img = vid.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        
        #print("shape",np.shape(img[y:y+h, x:x+w]))
        image = cv2.resize(img[y:y+h, x:x+w], (size, size))
        imagearray = img_to_array(image)
        data=np.expand_dims(imagearray, axis=0)
        result = model.predict(data)
        r=np.argmax(result)
        text = label[r]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255) if str(text) == "no_glasses" else (0,255,0), 2)
        draw_text(img,str(text), font_scale=2, pos=(x,y-5), 
                  text_color=(0,0,0), text_color_bg=(0, 0, 255) if str(text) == "no_glasses" else (0,255,0))
        #cv2.putText(img,str(text),(x,y-5), font, .5,(0,0,255) if str(text) == "no_glass" else (0,255,0) ,1, cv2.LINE_AA)
        
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27: #press escape to quite
        break
# Release the VideoCapture object
vid.release()
cv2.destroyAllWindows()


