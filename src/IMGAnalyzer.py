import cv2
from pathlib import Path
from PIL import Image , ImageDraw
import numpy as np 
import sys, os


class IMGAnalyzer():

    def __init__(self, image_path):
        
        self.image_path = image_path
        self.image = Image.open(self.image_path)

        
    def getFaces(self):
        
        image = Image.open(self.image_path)
        
        # load pre trained model
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    

        grayscale = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
        
        faces = classifier.detectMultiScale(grayscale, 1.3, 5)
        
        return faces

    def hasFaces(self):
        return self.getFaces > 0

    def drawRectangle(self, faces ):
        for (x, y, w, h) in faces:

            cv2.rectangle(np.asarray(self.image), (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        draw = ImageDraw.Draw(self.image)
        draw.rectangle([(x,y),(x+w,y+h)])
        self.image.show()
    
    def crop(self, faces):
        # assert len(faces)== 1, f"more than one face was detected "
        x, y, w ,h = (212, 167, 218,218)
        image_numpy = np.asarray(self.image)[y:y+h, x:x+w] 
        
        return image_numpy
        