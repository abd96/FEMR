import cv2
from pathlib import Path
from PIL import Image , ImageDraw
import numpy as np 
import sys, os
from imutils import face_utils
import imutils
import dlib 
np.set_printoptions(threshold=sys.maxsize)

class IMGAnalyzer():

    def __init__(self, image_path):
        
        self.image_path = image_path
        self.image = Image.open(self.image_path)
        
        self.facial_landmarks = {
            "mouth" : (48, 68),
            "right_eyebrow": (17, 22),
            "left_eyebrow": (22, 27),
            "right_eye": (36, 42),
            "left_eye": (42, 48),
            #"nose":(27, 35),
            "jaw": (0, 17)
        }
        
    def getFaces(self):
        
        image = Image.open(self.image_path)
        
        # load pre trained model
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    

        grayscale = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
        
        faces = classifier.detectMultiScale(grayscale, 1.3, 5)
        
        return faces

    def hasFaces(self):
        return self.getFaces > 0

    def drawRectangle(self):
        faces = self.getFaces()
        for (x, y, w, h) in faces:

            cv2.rectangle(np.asarray(self.image), (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        draw = ImageDraw.Draw(self.image)
        draw.rectangle([(x,y),(x+w,y+h)])
        self.image.show()
    
    def crop(self, face_rect):
        
        x ,y, h, w = face_rect
        return np.asarray(self.image)[y:y+h, x:x+w]
        
    def getFeatures(self, crop=False):
        
        if crop:
            self.image = Image.fromarray(self.crop(self.getFaces()[0]))
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join('model', 'shape_predictor_68_face_landmarks.dat'))
        gray_image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_BGR2GRAY)
        faces = detector(gray_image, 1)
        nump_image = np.asarray(self.image)
        for (i, face) in enumerate(faces):
            shape = predictor(gray_image, face)
            facial_landmarks = face_utils.shape_to_np(shape)
            for (name, (i, j)) in self.facial_landmarks.items():
                if name == 'mouth':
                    (x,y, w, h) = cv2.boundingRect(np.array([facial_landmarks[i:j]]))
                    landmark = nump_image[y:y + h, x:x + w]
                    landmark = imutils.resize(landmark, width=128, inter=cv2.INTER_CUBIC)
                    # Image.fromarray(landmark).show()
                    
                    
                    landmark = cv2.resize(landmark, (250, 250))
                    return landmark
                    
            
            # self.image.show()
    
    def getFeatures_mouth(self, crop=False):
        
        if crop:
            self.image = Image.fromarray(self.crop(self.getFaces()[0]))
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join('model', 'shape_predictor_68_face_landmarks.dat'))
        gray_image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_BGR2GRAY)
        faces = detector(gray_image, 1)
        face = faces[0]
        nump_image = np.asarray(self.image)
        shape = predictor(gray_image, face)
        facial_landmarks = face_utils.shape_to_np(shape)
        i, j = self.facial_landmarks['mouth']
        (x,y, w, h) = cv2.boundingRect(np.array([facial_landmarks[i:j]]))
        landmark = nump_image[y:y + h, x:x + w]
        landmark = imutils.resize(landmark, width=250, inter=cv2.INTER_CUBIC)
        landmark = cv2.resize(landmark, (250, 250))
        
        return landmark
    
    def load(self, path):
        # thanks to https://github.com/albanie
        """takes as input the path to a .pts and returns a list of 
        tuples of floats containing the points in in the form:
        [(x_0, y_0, z_0),
        (x_1, y_1, z_1),
        ...
        (x_n, y_n, z_n)]"""
        with open(path) as f:
            rows = [rows.strip() for rows in f]
        
        """Use the curly braces to find the start and end of the point data""" 
        head = rows.index('{') + 1
        tail = rows.index('}')

        """Select the point data split into coordinates"""
        raw_points = rows[head:tail]
        coords_set = [point.split() for point in raw_points]

        """Convert entries from lists of strings to tuples of floats"""
        points = [tuple([float(point) for point in coords]) for coords in coords_set]
        return points
