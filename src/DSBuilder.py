import os, sys
import pandas
from os.path import isfile, join
from pathlib import Path
from PIL import Image , ImageDraw
import pandas as pd
import numpy as  np
import csv 
import cv2
from keras.preprocessing.image import *
from src.IMGAnalyzer import IMGAnalyzer
from tqdm import tqdm 

np.set_printoptions(threshold=sys.maxsize)
class DSBuilder():
    
    def __init__(self):
        self.path = Path(os.getcwd(), 'FDB', 'facesdb')
        self.labelMap = {
            # 0 : 'normal',
            1 : 'happy',
            2 : 'sad',
            # 3 : 'surprised',
            4 : 'angry',
            # 5 : 'disgust', 
            6 : 'fear'
        }
        # pandas dataframe to find the labeld data to -> used to traing model
        self.dataset = []
        # self.dataset = pd.DataFrame(columns = ['image', 0, 1, 2, 3, 4, 5, 6])
        

    def seek(self):
        dimension_row = 0
        # works only on the database saved on my pc
        # modeify if you need to train on another dataset   
        # for every subdirectory need to get subdirs that has images 
        
        for subdir, dir, files in tqdm(os.walk(self.path)):
            
            # len =9 -> (for my dataset) where the images are saved 
            if len(subdir.split("/")) == 10:
                # iterate over files and choose the 6 files 
                # that corresponds to the expression defined
                # above in labelMap
                for  dir, _, files in os.walk(subdir):
                    if files:
                        
                        if dir.endswith("bmp"):
                            
                            for i in [1,2, 4,6]:
                               
                                proband, filename = files[i].split("-")
                                
                                image_path = Path(dir, proband+"-"+filename)
                                # image = Image.open(image_path)
                                # image_nump = np.array(image)
                                
                                # analyzer for cropping images see -> IMGAnalyzer.py
                                analyzer = IMGAnalyzer(image_path)
                                
                                
                                image_nump = analyzer.crop(analyzer.getFaces()[0])
                                
                                # Image.fromarray(image_nump).show()
                                image_nump = cv2.resize(image_nump, (128, 128))
                               
                                # print(image_nump.shape)

                                # normalize image data 
                                
                                
                                
                                # image = load_img(Path(dir, proband+"-"+filename))
                                # image_nump = img_to_array(image)
                                
                                
                                # img = Image.fromarray(image_nump)
                                # img.show()
                                
                                # create label vector
                                #
                                label = np.zeros(7)
                                label[int(filename.split("_")[0])] = 1
                                
                                label = [label[1] , label[2], label[4], label[6]]
                                
                                # print(image_nump)
                                
                                # convert image to one channel -> grey color no rgb(3 channels) -> new shape (480, 640)
                                # image_nump = cv2.cvtColor(image_nump, cv2.COLOR_BGR2GRAY)
                                
                                # normalize data 
                                
                                
                                self.dataset.append((files[i].split(".")[0], image_nump/255.0, label))
                                
                                # create a sample 
                                # concatenated with labels -> last 7 indicies are the 7 classes(expressions)
                                
                                # sample = np.concatenate((image_nump.flatten('F'), label))
                                
                                dimension_row += 1
                                # self.exportData(sample)
                                

                        # print(dimension_row)
                        # numpy.zeros(dimension_row, ((480, 640, 3), 8))
                
    def exportData(self, sample):
        with open('dataset.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer = writer.writerow(sample)

                