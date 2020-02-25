import cv2
from pathlib import Path
from PIL import Image , ImageDraw
import numpy as np 
import sys, os


class Detector():

    def __init__(self):
        self.image = Path(os.getcwd(), 'FDB', 'facesdb', 's001', 'bmp', 's001-00_img.bmp')
        

    def getFaces(self):
        
        image = Image.open(self.image)
        
        # load pre trained model
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    

        grayscale = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
        
        faces = classifier.detectMultiScale(grayscale, 1.3, 5)

        return faces

    def hasFaces(self):
        return self.getFaces > 0

    def drawRectangle(self, image, faces ):
        for (x, y, w, h) in faces:
            cv2.rectangle(np.asarray(image), (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        draw = ImageDraw.Draw(image)
        draw.rectangle([(x,y),(x+w,y+h)])
        image.show()
    