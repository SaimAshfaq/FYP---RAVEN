from __future__ import division, print_function, absolute_import
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QPixmap
from yolo_v3 import YOLO3
#from yolo_v4 import YOLO4
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from reid import REID
#import tkinter as tk
from tkinter import filedialog
from urllib import parse
from timeit import time
from collections import Counter
from home import Ui_MainWindow
from datetime import datetime

import sys
import numpy as np
from tkinter import filedialog
import tkinter as tk
import cv2
import os
import glob
import sqlite3
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import warnings
import argparse
import base64
import requests
import urllib

import json
import random
import time
from PIL import Image
import operator


parser = argparse.ArgumentParser()
parser.add_argument('-all', help='Combine all videos into one', default=True)
args = parser.parse_args()

class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if not os.path.isfile(path):
            raise FileExistsError
        
        self.cap = cv2.VideoCapture(path)        
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        print('Length of {}: {:d} frames'.format(path,self.vn))

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh

def play_videoFile(filePath,mirror=False):
    slowmo = 60
    cap = cv2.VideoCapture(filePath)
    #cv2.namedWindow('Video ',cv2.WINDOW_AUTOSIZE)
    while True:
        ret_val, frame = cap.read()
        if mirror:
            frame = cv2.flip(frame, 1)
        cv2.imshow('Output Video', frame)
        if cv2.waitKey(slowmo) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
    
def displayImage(self, img,window=1):
    qformat = QImage.Format_Indexed8
    if len(img.shape) == 3:
        if(img.shape[2]) ==4:
            qformat = QImage.Format_RGBA888
            
        else:
            qformat = QImage.Format_RGB888
            
    img = QImage(img, img.shape[1], img.shape[0], qformat)
    img = img.rgbSwapped()
    self.imgLabel.setPixmap(QPixmap.fromImage(img))
    
class WelcomeScreen(QDialog):
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        loadUi("welcomescreen.ui",self)
        self.login.clicked.connect(self.gotologin)
        self.create.clicked.connect(self.gotocreate)
    
    def gotologin(self):
        login = LoginScreen()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def gotocreate(self):
        create = CreateAccScreen()
        widget.addWidget(create)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
class Report(QMainWindow):
    def __init__(self):
       super(Report, self).__init__()
       loadUi("report.ui",self)
       self.tableWidget.setColumnWidth(0,135)
       self.tableWidget.setColumnWidth(1,330)
       #self.tableWidget.setColumnWidth(2,235)
       self.tableWidget.setHorizontalHeaderLabels(["Id", "Appearance Time"])
       self.search_button_2.clicked.connect(self.View_Report_byid)
       self.rep_btn.clicked.connect(self.View_Full_Report)
       self.switch_page.clicked.connect(self.gotoHome)
       
    def View_Full_Report(self):
        conn = sqlite3.connect("Raven_DB.db")
        cur = conn.cursor()
        query = "SELECT DISTINCT Id,Time FROM Report where Id in (SELECT DISTINCT Id from Report)"
        
        #rowCount = "SELECT COUNT(*) FROM (SELECT DISTINCT Id,Time FROM Report where Id in (SELECT DISTINCT Id from Report))"
        #count = cur.execute(rowCount)
        #count = cur.count()
        
        self.tableWidget.setRowCount(100)
        tableRow = 0
        
        for row in cur.execute(query):
            self.error_2.setText("")
            print(row)
            self.tableWidget.setItem(tableRow, 0, QtWidgets.QTableWidgetItem(str(row[0])))
            self.tableWidget.setItem(tableRow, 1, QtWidgets.QTableWidgetItem(row[1]))
            tableRow=tableRow+1
        
        conn.commit()
        cur.close()
        conn.close()
      
    def View_Report_byid(self):
        inp_id = self.SearchEdit2.text()
        print(inp_id)
        conn = sqlite3.connect("Raven_DB.db")
        cur = conn.cursor()
        print("Successfully Connected to SQLite")
        query = "SELECT DISTINCT Id,Time FROM Report where Id={i}".format(i=inp_id)
        query2 = "SELECT Id FROM Report where Id={i}".format(i=inp_id)
        cur.execute(query2)
        row1 = cur.fetchone()
        if row1 is None:
            self.error_2.setText("No person with that ID exists in our system.")
            self.tableWidget.setRowCount(0)
        else:
            self.tableWidget.setRowCount(100)
            tableRow = 0
            for row in cur.execute(query):
            #if row is None:
             #   self.error_2.setText("No person with that ID exists in our system.")
                self.error_2.setText("")
                print(row)
                self.tableWidget.setItem(tableRow, 0, QtWidgets.QTableWidgetItem(str(row[0])))
                self.tableWidget.setItem(tableRow, 1, QtWidgets.QTableWidgetItem(row[1]))
            #self.tableWidget.setItem(tableRow, 2, QtWidgets.QTableWidgetItem(row[2]))
                tableRow=tableRow+1
        #cur.execute(query)
        #row_check = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
    
    def gotoHome(self):
        window = MainWindow()
        #window = FrontEnd()
        widget.addWidget(window)
        #widget.setFixedHeight(700)
        #widget.setFixedWidth(750)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
        
class ViewVideo(QMainWindow):
    def __init__(self):
       super(ViewVideo, self).__init__()
       loadUi("viewVideo.ui",self)
       self.rep_btn.clicked.connect(self.search_video_byid)
       self.switch_page.clicked.connect(self.gotoHome)
       
       
#       =================================== Search Video By ID ================================ 
       
    def search_video_byid(self):
        input_id = self.SearchEdit2.text()
        print(input_id)
        self.videoArea.setText("")
        conn = sqlite3.connect("Raven_DB.db")
        cur = conn.cursor()
        print("Successfully Connected to SQLite")
        outp_dir = '/videos/output/tracklets/{s}.avi'.format(s=str(input_id))
        print(outp_dir)
        query = "SELECT IdPath from Tracklets where Id={i}".format(i=input_id)
        print('Query: ', query)
        cur.execute(query)
        row = cur.fetchone()
        print(row)
        #print(idx)
        
        if input_id is None or row is None:   
            self.search_error.setText("No person with that ID exists in our system.")
            self.videoArea.setText(".")
        
        else:
            self.search_error.setText("")
            l = list(row)
            final_path = "".join(l)
            print('Final result', final_path)
            init_path = 'C:/Users/DELL/Desktop/Multi-Camera-Person-Tracking-and-Re-Identification'
            complete_path = init_path + final_path
            print(complete_path)
            
            cap = cv2.VideoCapture(complete_path)
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    self.showVideo(frame,1)
                if cv2.waitKey(60) == 27:
                        break
            capture.release()
            cv2.destroyAllWindows()
            cv2.destroyAllWindows()
            
            conn.commit()
            cur.close()
            conn.close()
            
            
    def showVideo(self, img,window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if(img.shape[2]) ==4:
                qformat = QImage.Format_RGBA888 
                
            else:
                qformat = QImage.Format_RGB888           
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.videoArea.setPixmap(QPixmap.fromImage(img))
    
    def gotoHome(self):
        window = MainWindow()
        #window = FrontEnd()
        widget.addWidget(window)
        #widget.setFixedHeight(700)
        #widget.setFixedWidth(750)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
class CreateAccScreen(QDialog):
    def __init__(self):
        super(CreateAccScreen, self).__init__()
        loadUi("createacc.ui",self)
        self.passwordfield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirmpasswordfield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.signup.clicked.connect(self.signupfunction)
        self.switch_page.clicked.connect(self.gotoWelcome)
    
    def signupfunction(self):
        user = self.emailfield.text()
        password = self.passwordfield.text()
        confirmpassword = self.confirmpasswordfield.text()

        if len(user)==0 or len(password)==0 or len(confirmpassword)==0:
            self.error.setText("Please fill in all inputs.")

        elif password!=confirmpassword:
            self.error.setText("Passwords do not match.")
            
        else:
            conn = sqlite3.connect("Raven_DB.db")
            cur = conn.cursor()

            user_info = [user, password]
            query = "SELECT user from login_info where user like '%{a}%'".format(a=user_info[0])
            cur.execute(query)
            result_pass = cur.fetchone()
            
            if result_pass==None:
               cur.execute('INSERT INTO login_info (user, password) VALUES (?,?)', user_info)
               login = LoginScreen()
               widget.addWidget(login)
               #widget.setFixedWidth(750)
               #widget.setFixedHeight(700)
               widget.setCurrentIndex(widget.currentIndex()+1)
            else:
                self.error.setText("User with that username already exists.")

            conn.commit()
            conn.close()
            
    def gotoWelcome(self):
        welcome = WelcomeScreen()
        widget.addWidget(welcome)
        #widget.setFixedHeight(700)
        #widget.setFixedWidth(750)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
class LoginScreen(QDialog):
    def __init__(self):
        super(LoginScreen, self).__init__()
        loadUi("login.ui",self)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.login.clicked.connect(self.loginfunction)
        self.switch_page.clicked.connect(self.gotoWelcome)

    '''def signupfunction(self):
        user = self.emailfield.text()
        password = self.passwordfield.text()
        confirmpassword = self.confirmpasswordfield.text()

        if len(user)==0 or len(password)==0 or len(confirmpassword)==0:
            self.error.setText("Please fill in all inputs.")

        elif password!=confirmpassword:
            self.error.setText("Passwords do not match.")
        else:
            conn = sqlite3.connect("Raven_DB.db")
            cur = conn.cursor()

            user_info = [user, password]
            cur.execute('INSERT INTO login_info (user, password) VALUES (?,?)', user_info)

            conn.commit()
            print('New user entered successfully')
            conn.close()

            login = LoginScreen()
            widget.addWidget(login)
            widget.setFixedWidth(700)
            widget.setFixedHeight(600)
            widget.setCurrentIndex(widget.currentIndex()-1)'''
    
    def loginfunction(self):
        user = self.email.text()
        password = self.password.text()
        

        if len(user)==0:
            self.error.setText("Username cannot be empty")
        elif len(password)==0:
            self.error.setText("Password cannot be empty")
        elif len(user)==0 and len(password)==0:
            self.error.setText("Username and Password cannot be empty")

        else:
            conn = sqlite3.connect("Raven_DB.db")
            cur = conn.cursor()
            #query = 'SELECT Password FROM login_info WHERE Username =\''+user+"\'"
            query = "SELECT user,password from login_info WHERE user like '"+user + "'and password like '" + password + "'"
            cur.execute(query)
            #result_pass = cur.fetchone()[0]
            result_pass = cur.fetchone()
            if result_pass == None:
                #self.success.setText("")
                self.error.setText("Incorrect username or password")
            else:
                self.error.setText("")
                user = self.email.setText('')
                password = self.password.setText('')
                #self.success.setText("Logged in Successfully!")
                self.gotoHome()
                print("Successfully logged in.") 

    def gotoHome(self):
        window = MainWindow()
        #window = FrontEnd()
        widget.addWidget(window)
        #widget.setFixedHeight(700)
        #widget.setFixedWidth(750)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def gotoWelcome(self):
        welcome = WelcomeScreen()
        widget.addWidget(welcome)
        #widget.setFixedHeight(700)
        #widget.setFixedWidth(750)
        widget.setCurrentIndex(widget.currentIndex()+1)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.webcam_btn.clicked.connect(self.webcam)
        self.dircam_btn.clicked.connect(self.fromGallery)
        self.report_btn.clicked.connect(self.gotoReport)
        self.video_btn.clicked.connect(self.gotoViewVideo)
        #self.search_button.clicked.connect(self.search_video_byid)
        self.logout_btn.clicked.connect(self.gotoLogin)
        self.setWindowTitle('Welcome to RAVEN')
        #self.resize(1080,820)

#       ========================================= LogOut =====================================
    def gotoLogin(self):
        login = LoginScreen()
        widget.addWidget(login)
        #widget.setFixedWidth(750)
        #widget.setFixedHeight(700)
        widget.setCurrentIndex(widget.currentIndex()+1)   

    
#       ========================================= View Report =====================================            
    
    def gotoReport(self):
        report = Report()
        widget.addWidget(report)
        #widget.setFixedWidth(750)
        #widget.setFixedHeight(700)
        widget.setCurrentIndex(widget.currentIndex()+1)  
        
        
#       ========================================= View Video =====================================            
    
    def gotoViewVideo(self):
        viewVideo = ViewVideo()
        widget.addWidget(viewVideo)
        #widget.setFixedWidth(750)
        #widget.setFixedHeight(700)
        widget.setCurrentIndex(widget.currentIndex()+1)  
    
    
            
#       =================================== Display Video in Home ================================  

        
    def displayImage(self, img,window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if(img.shape[2]) ==4:
                qformat = QImage.Format_RGBA888 
                
            else:
                qformat = QImage.Format_RGB888
                
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.imgLabel.setPixmap(QPixmap.fromImage(img))
        

#        ==========================================web images==================================
    def webcam(self):
        capture = cv2.VideoCapture(0)
         
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        videoWriter = cv2.VideoWriter('C:/Users/DELL/Desktop/Multi-Camera-Person-Tracking-and-Re-Identification/WebCamVid.avi', fourcc, 30.0, (640,480))
         
        while (True):
         
            ret, frame = capture.read()
             
            if ret:
                cv2.imshow('video', frame)
                videoWriter.write(frame)
         
            if cv2.waitKey(1) == 27:
                break
         
        capture.release()
        videoWriter.release()
         
        cv2.destroyAllWindows()
        
        def main(yolo):
            
             #WebCamVideo = 'C:/Users/DELL/Desktop/FYP_Integration/WebCamVid.avi'
             webcamvideo = "C:/Users/DELL/Desktop/Multi-Camera-Person-Tracking-and-Re-Identification/WebCamVid.avi"

             
             videolist = [webcamvideo]
             
             max_cosine_distance = 0.2
             nn_budget = None
             nms_max_overlap = 0.4
             
            # deep_sort 
             model_filename = 'model_data/models/mars-small128.pb'
             encoder = gdet.create_box_encoder(model_filename,batch_size=1) # use to get feature
             
             metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
             tracker = Tracker(metric, max_age=100)

             is_vis = True
             out_dir = 'videos/output/'
             print('The output folder is',out_dir)
             if not os.path.exists(out_dir):
                 os.mkdir(out_dir)

             all_frames = []
             for video in videolist:
                 print("This is video path",video)
                 loadvideo = LoadVideo(video)
                 video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
                 while True:
                     ret, frame = video_capture.read() 
                     if ret != True:
                         video_capture.release()
                         break
                     all_frames.append(frame)

             frame_nums = len(all_frames)
             tracking_path = out_dir+'tracking'+'.avi'
             combined_path = out_dir+'allVideos'+'.avi'
             if is_vis:
                 fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                 out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
                 out2 = cv2.VideoWriter(combined_path, fourcc, frame_rate, (w, h))
                 #Combine all videos
                 for frame in all_frames:
                     out2.write(frame)
                 out2.release()
                 
             #Initialize tracking file
             filename = out_dir+'/tracking.txt'
             open(filename, 'w')
             
             #Initialize reid file
             reid_filename = out_dir+'/reid.txt'
             open(reid_filename, 'w')
             
            # fps = 0.0
             frame_cnt = 0
             t1 = time.time()
             
             track_cnt = dict()
             images_by_id = dict()
             ids_per_frame = []
             for frame in all_frames:
                 image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                 boxs = yolo.detect_image(image) # n * [topleft_x, topleft_y, w, h]
                 features = encoder(frame,boxs) # n * 128
                 detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)] # length = n
                 text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

                 
                 # Run non-maxima suppression.
                 boxes = np.array([d.tlwh for d in detections])
                 scores = np.array([d.confidence for d in detections])
                 indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores) #preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                 detections = [detections[i] for i in indices] # length = len(indices)

                 # Call the tracker 
                 tracker.predict()
                 tracker.update(detections)
                 tmp_ids = []
                 for track in tracker.tracks:
                     if not track.is_confirmed() or track.time_since_update > 1:
                         continue 
                     bbox = track.to_tlbr()
                     area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
                     if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                         tmp_ids.append(track.track_id)
                         if track.track_id not in track_cnt:
                             track_cnt[track.track_id] = [[frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                             images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                         else:
                             track_cnt[track.track_id].append([frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area])
                             images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                     cv2_addBox(track.track_id,frame,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),line_thickness,text_thickness,text_scale)
                     curr_time = datetime.now()
                     dt = curr_time.strftime("%d/%m/%Y %H:%M:%S")
                     print("date and time =", dt)
                     write_results(filename,'mot',frame_cnt+1,str(track.track_id), dt, int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),w,h)
                 ids_per_frame.append(set(tmp_ids))

                 # save a frame               
                 if is_vis:
                     out.write(frame)
                 t2 = time.time()
                 
                 frame_cnt += 1
                 print(frame_cnt, '/', frame_nums)

             if is_vis:
                 out.release()
             print('Tracking finished in {} seconds'.format(int(time.time() - t1)))
             print('Tracked video : {}'.format(tracking_path))
             print('Combined video : {}'.format(combined_path))

             os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
             reid = REID()
             threshold = 320
             exist_ids = set()
             final_fuse_id = dict()

             print('Total IDs = ',len(images_by_id))
             feats = dict()
             for i in images_by_id:
                 print('ID number {} -> Number of frames {}'.format(i, len(images_by_id[i])))
                 feats[i] = reid._features(images_by_id[i]) 
             
             for f in ids_per_frame:
                 if f:
                     if len(exist_ids) == 0:
                         for i in f:
                             final_fuse_id[i] = [i]
                         exist_ids = exist_ids or f
                     else:
                         new_ids = f-exist_ids
                         for nid in new_ids:
                             dis = []
                             if len(images_by_id[nid])<10:
                                 exist_ids.add(nid)
                                 continue
                             unpickable = []
                             for i in f:
                                 for key,item in final_fuse_id.items():
                                     if i in item:
                                         unpickable += final_fuse_id[key]
                             print('exist_ids {} unpickable {}'.format(exist_ids,unpickable))
                             for oid in (exist_ids-set(unpickable))&set(final_fuse_id.keys()):
                                 tmp = np.mean(reid.compute_distance(feats[nid],feats[oid]))
                                 print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                                 dis.append([oid, tmp])
                             exist_ids.add(nid)
                             if not dis:
                                 final_fuse_id[nid] = [nid]
                                 continue
                             dis.sort(key=operator.itemgetter(1))
                             if dis[0][1] < threshold:
                                 combined_id = dis[0][0]
                                 images_by_id[combined_id] += images_by_id[nid]
                                 final_fuse_id[combined_id].append(nid)
                             else:
                                 final_fuse_id[nid] = [nid]
             print('Final ids and their sub-ids:',final_fuse_id)
             print('MOT took {} seconds'.format(int(time.time() - t1)))
             #print('final_fuse_id', final)
             t2 = time.time()

             # To generate MOT for each person, declare 'is_vis' to True
             is_vis=True
             if is_vis:
                 print('Writing videos for each ID...')
                 output_dir = 'videos/output/tracklets/'
                 if not os.path.exists(output_dir):
                     os.mkdir(output_dir)
                 loadvideo = LoadVideo(combined_path)
                 video_capture,frame_rate, w, h = loadvideo.get_VideoLabels()
                 fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                 for idx in final_fuse_id:
                     tracking_path = os.path.join(output_dir, str(idx)+'.avi')
                     out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
                     for i in final_fuse_id[idx]:
                         for f in track_cnt[i]:
                             video_capture.set(cv2.CAP_PROP_POS_FRAMES, f[0])
                             _, frame = video_capture.read()
                             text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                             cv2_addBox(idx, frame, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                             out.write(frame)
                     out.release()
                 video_capture.release()
            
             for idx in final_fuse_id:
                 print('final ids: ', str(idx) )
                 outp_dir = "videos/output/tracklets/{s}.avi".format(s=str(idx))
                 print("Path:",outp_dir)
             #print('final id:', final_fuse_id)

             # Generate a single video with complete MOT/ReID              
             if args.all:
                 loadvideo = LoadVideo(combined_path)
                 video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
                 fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                 complete_path = out_dir+'/Complete'+'.avi'
                 out = cv2.VideoWriter(complete_path, fourcc, frame_rate, (w, h))
                 
                 for frame in range(len(all_frames)):
                     frame2 = all_frames[frame]
                     video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                     _, frame2 = video_capture.read()
                     for idx in final_fuse_id:
                         for i in final_fuse_id[idx]:
                             for f in track_cnt[i]:
                                 #print('frame {} f0 {}'.format(frame,f[0]))
                                 if frame == f[0]:
                                     text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
                                     cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                             curr_time = datetime.now()
                             dt = curr_time.strftime("%d/%m/%Y %H:%M:%S")
                             conn = sqlite3.connect("Raven_DB.db")
                             cur = conn.cursor()
                             report_info = [idx, dt]
                             query = "SELECT Id from Report where Id like '%{r}%'".format(r=report_info[0])
                             cur.execute(query)
                             idCheck = cur.fetchone()
                             if idCheck==None:
                                cur.execute("INSERT INTO Report (Id, Time) VALUES ({x}, '{t}')".format(x=idx,t=str(dt)))
                             else:
                                cur.execute("INSERT INTO Report (Id, Time, Reappeared) VALUES ({x}, '{t}', 'YES')".format(x=idx,t=str(dt)))
                                
                             conn.commit()
                             cur.close()
                             conn.close()
                             write_results_after_reid(reid_filename,'mot', idx, dt) 
                     out.write(frame2)
                 out.release()
                 video_capture.release()
                 
                 for idx in final_fuse_id:
                     outp_dir = '/videos/output/tracklets/{s}.avi'.format(s=str(idx))
                     print(outp_dir)
                     print(idx)
                     conn = sqlite3.connect("Raven_DB.db")
                     cur = conn.cursor()
                     print("Successfully Connected to SQLite")
                     tracklet_info = [idx, outp_dir]
                     query = "SELECT Id from Tracklets where Id like '%{a}%'".format(a=tracklet_info[0])
                     cur.execute(query)
                     check = cur.fetchone()
                     if check==None:
                        cur.execute("INSERT INTO Tracklets (Id, IdPath) VALUES ({i}, '{o}')".format(i=idx,o=outp_dir))
                     #else:
                      #   cur.execute("UPDATE Tracklets set Id={i}, IdPath={o}".format(i=idx,o=outp_dir))
                     
                     conn.commit()
                     cur.close()
                     conn.close()

             #os.remove(combined_path)
             print('\nWriting videos took {} seconds'.format(int(time.time() - t2)))
             print('Final video at {}'.format(complete_path))
             print('Total: {} seconds'.format(int(time.time() - t1)))
             capture = cv2.VideoCapture("C:/Users/DELL/Desktop/Multi-Camera-Person-Tracking-and-Re-Identification/videos/output/Complete.avi")
             #capture = cv2.VideoCapture(0)
             while (capture.isOpened()):
                 ret, frame = capture.read()
                 if ret == True:
                     self.displayImage(frame,1)
                 if cv2.waitKey(60) == 27:
                         break
             capture.release()
             cv2.destroyAllWindows()
             #play_videoFile(complete_path,mirror=False)
        
        if __name__ == '__main__':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
            main(yolo= YOLO3())
            #main(yolo= YOLO4()) 
        
#====================================================fromdirectory==========================
    def fromGallery(self):
        
        def main(yolo):
                           
             testvideo,_ = QFileDialog.getOpenFileName(self, 'Select Video', QDir.currentPath(), 'Videos (*.mp4 *.avi)')
             #testvideo2 = filedialog.askopenfilename(title='open')

             #videolist = [testvideo, testvideo2]
             videolist = [testvideo]
             
             #root.mainloop()
             
             max_cosine_distance = 0.2
             nn_budget = None
             nms_max_overlap = 0.4
             
            # deep_sort 
             model_filename = 'model_data/models/mars-small128.pb'
             encoder = gdet.create_box_encoder(model_filename,batch_size=1) # use to get feature
             
             metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
             tracker = Tracker(metric, max_age=100)

             is_vis = True
             out_dir = 'videos/output/'
             print('The output folder is',out_dir)
             if not os.path.exists(out_dir):
                 os.mkdir(out_dir)

             all_frames = []
             for video in videolist:
                 print("This is video path",video)
                 loadvideo = LoadVideo(video)
                 video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
                 while True:
                     ret, frame = video_capture.read() 
                     if ret != True:
                         video_capture.release()
                         break
                     all_frames.append(frame)

             frame_nums = len(all_frames)
             tracking_path = out_dir+'tracking'+'.avi'
             combined_path = out_dir+'allVideos'+'.avi'
             if is_vis:
                 fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                 out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
                 out2 = cv2.VideoWriter(combined_path, fourcc, frame_rate, (w, h))
                 #Combine all videos
                 for frame in all_frames:
                     out2.write(frame)
                 out2.release()
                 
             #Initialize tracking file
             filename = out_dir+'/tracking.txt'
             open(filename, 'w')
             
             #Initializing reid file
             reid_filename = out_dir+'/reid.txt'
             open(reid_filename, 'w')
             
            # fps = 0.0
             frame_cnt = 0
             t1 = time.time()
             
             track_cnt = dict()
             images_by_id = dict()
             ids_per_frame = []
             for frame in all_frames:
                 image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                 boxs = yolo.detect_image(image) # n * [topleft_x, topleft_y, w, h]
                 features = encoder(frame,boxs) # n * 128
                 detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)] # length = n
                 text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

                 
                 # Run non-maxima suppression.
                 boxes = np.array([d.tlwh for d in detections])
                 scores = np.array([d.confidence for d in detections])
                 indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores) #preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                 detections = [detections[i] for i in indices] # length = len(indices)

                 # Call the tracker 
                 tracker.predict()
                 tracker.update(detections)
                 tmp_ids = []
                 for track in tracker.tracks:
                     if not track.is_confirmed() or track.time_since_update > 1:
                         continue 
                     
                     bbox = track.to_tlbr()
                     area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
                     if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                         tmp_ids.append(track.track_id)
                         if track.track_id not in track_cnt:
                             track_cnt[track.track_id] = [[frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                             images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                         else:
                             track_cnt[track.track_id].append([frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area])
                             images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                     cv2_addBox(track.track_id,frame,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),line_thickness,text_thickness,text_scale)
                     curr_time = datetime.now()
                     dt = curr_time.strftime("%d/%m/%Y %H:%M:%S")
                     print("date and time =", dt)
                     write_results(filename,'mot',frame_cnt+1,str(track.track_id), dt, int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),w,h)
                 ids_per_frame.append(set(tmp_ids))

                 # save a frame               
                 if is_vis:
                     out.write(frame)
                 t2 = time.time()
                 
                 frame_cnt += 1
                 print(frame_cnt, '/', frame_nums)

             if is_vis:
                 out.release()
             print('Tracking finished in {} seconds'.format(int(time.time() - t1)))
             print('Tracked video : {}'.format(tracking_path))
             print('Combined video : {}'.format(combined_path))

             os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
             reid = REID()
             threshold = 320
             #threshold = 450
             exist_ids = set()
             final_fuse_id = dict()

             print('Total IDs = ',len(images_by_id))
             feats = dict()
             for i in images_by_id:
                 print('ID number {} -> Number of frames {}'.format(i, len(images_by_id[i])))
                 feats[i] = reid._features(images_by_id[i]) 
             
             for f in ids_per_frame:
                 if f:
                     if len(exist_ids) == 0:
                         for i in f:
                             final_fuse_id[i] = [i]
                         exist_ids = exist_ids or f
                     else:
                         new_ids = f-exist_ids
                         for nid in new_ids:
                             dis = []
                             if len(images_by_id[nid])<10:
                                 exist_ids.add(nid)
                                 continue
                             unpickable = []
                             for i in f:
                                 for key,item in final_fuse_id.items():
                                     if i in item:
                                         unpickable += final_fuse_id[key]
                             print('exist_ids {} unpickable {}'.format(exist_ids,unpickable))
                             for oid in (exist_ids-set(unpickable))&set(final_fuse_id.keys()):
                                 tmp = np.mean(reid.compute_distance(feats[nid],feats[oid]))
                                 print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                                 dis.append([oid, tmp])
                             exist_ids.add(nid)
                             if not dis:
                                 final_fuse_id[nid] = [nid]
                                 continue
                             dis.sort(key=operator.itemgetter(1))
                             if dis[0][1] < threshold:
                                 combined_id = dis[0][0]
                                 print('dis[0][0]: ', dis[0][0])
                                 print('dis[0][1]: ', dis[0][1])
                                 #curr_time = datetime.now()
                                 #dt = curr_time.strftime("%d/%m/%Y %H:%M:%S")
                                 #write_results_after_reid(reid_filename,'mot',combined_id, dt)
                                 images_by_id[combined_id] += images_by_id[nid]
                                 final_fuse_id[combined_id].append(nid)
                             else:
                                 final_fuse_id[nid] = [nid]
                             #write_results_after_reid(reid_filename,'mot',final_fuse_id, dt)
             print('Final ids and their sub-ids:',final_fuse_id)
             print('MOT took {} seconds'.format(int(time.time() - t1)))
             t2 = time.time()
             #curr_time = datetime.now()
             #dt = curr_time.strftime("%d/%m/%Y %H:%M:%S")
             #print("date and time =", dt)
             #write_results_after_reid(reid_filename,'mot',final_fuse_id, dt)

             # To generate MOT for each person, declare 'is_vis' to True
             is_vis=True
             if is_vis:
                 print('Writing videos for each ID...')
                 output_dir = 'videos/output/tracklets/'
                 if not os.path.exists(output_dir):
                     os.mkdir(output_dir)
                 loadvideo = LoadVideo(combined_path)
                 video_capture,frame_rate, w, h = loadvideo.get_VideoLabels()
                 fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                 for idx in final_fuse_id:
                     tracking_path = os.path.join(output_dir, str(idx)+'.avi')
                     out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
                     for i in final_fuse_id[idx]:
                         for f in track_cnt[i]:
                             video_capture.set(cv2.CAP_PROP_POS_FRAMES, f[0])
                             _, frame = video_capture.read()
                             text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                             cv2_addBox(idx, frame, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                             out.write(frame)
                     out.release()
                 video_capture.release()
                 
                 for idx in final_fuse_id:
                     outp_dir = '/videos/output/tracklets/{s}.avi'.format(s=str(idx))
                     print(outp_dir)
                     print(idx)
                     con = sqlite3.connect("Raven_DB.db")
                     cur = con.cursor()
                     print("Successfully Connected to SQLite")
                     query = "INSERT INTO Tracklets (Id, IdPath) VALUES ({i}, '{o}')".format(i=idx,o=outp_dir)
                     cur.execute(query)
                     con.commit()
                     cur.close()
                     con.close()
                 
             # Generate a single video with complete MOT/ReID              
             if args.all:
                 loadvideo = LoadVideo(combined_path)
                 video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
                 fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                 complete_path = out_dir+'/Complete'+'.avi'
                 out = cv2.VideoWriter(complete_path, fourcc, frame_rate, (w, h))
                 
                 for frame in range(len(all_frames)):
                     frame2 = all_frames[frame]
                     #print('frame2: ', frame2)
                     video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                     _, frame2 = video_capture.read()
                     for idx in final_fuse_id:
                         for i in final_fuse_id[idx]:
                             for f in track_cnt[i]:
                                 #print('frame {} f0 {}'.format(frame,f[0]))
                                 if frame == f[0]:
                                     text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
                                     cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                             curr_time = datetime.now()
                             dt = curr_time.strftime("%d/%m/%Y %H:%M:%S")
                             conn = sqlite3.connect("Raven_DB.db")
                             cur = conn.cursor()
                             report_info = [idx, dt]
                             query = "SELECT Id from Report where Id like '%{r}%'".format(r=report_info[0])
                             cur.execute(query)
                             idCheck = cur.fetchone()
                             if idCheck==None:
                                cur.execute("INSERT INTO Report (Id, Time) VALUES ({x}, '{t}')".format(x=idx,t=str(dt)))
                             else:
                                cur.execute("INSERT INTO Report (Id, Time, Reappeared) VALUES ({x}, '{t}', 'YES')".format(x=idx,t=str(dt)))
                                
                             conn.commit()
                             cur.close()
                             conn.close()
                             write_results_after_reid(reid_filename,'mot', idx, dt) 
                             
                     out.write(frame2) 
                 out.release()
                 video_capture.release()
                 
                 for idx in final_fuse_id:
                     outp_dir = '/videos/output/tracklets/{s}.avi'.format(s=str(idx))
                     print(outp_dir)
                     print(idx)
                     conn = sqlite3.connect("Raven_DB.db")
                     cur = conn.cursor()
                     print("Successfully Connected to SQLite")
                     tracklet_info = [idx, outp_dir]
                     query = "SELECT Id from Tracklets where Id like '%{a}%'".format(a=tracklet_info[0])
                     cur.execute(query)
                     check = cur.fetchone()
                     if check==None:
                        cur.execute("INSERT INTO Tracklets (Id, IdPath) VALUES ({i}, '{o}')".format(i=idx,o=outp_dir))
                     #else:
                      #   cur.execute("UPDATE Tracklets set Id={i}, IdPath={o}".format(i=idx,o=outp_dir))
                        
                     conn.commit()
                     cur.close()
                     conn.close()

             os.remove(combined_path)
             print('\nWriting videos took {} seconds'.format(int(time.time() - t2)))
             print('Final video at {}'.format(complete_path))
             print('Total: {} seconds'.format(int(time.time() - t1)))
             capture = cv2.VideoCapture("C:/Users/DELL/Desktop/Multi-Camera-Person-Tracking-and-Re-Identification/videos/output/Complete.avi")
             #capture = cv2.VideoCapture(0)
             while (capture.isOpened()):
                 ret, frame = capture.read()
                 if ret == True:
                     self.displayImage(frame,1)
                 if cv2.waitKey(60) == 27:
                         break
             capture.release()
             cv2.destroyAllWindows()
             #play_videoFile(complete_path,mirror=False)
        
        if __name__ == '__main__':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
            main(yolo= YOLO3())
            #main(yolo= YOLO4()) 

def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness

def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness,text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2),color=color, thickness=line_thickness)
    cv2.putText(frame, str(track_id),(x1, y1+30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0,0,255),thickness=text_thickness)
 
def write_results_after_reid(reid_filename, data_type, w_reid, dt):
    if data_type == 'mot':
        save_format = '{id},{t}\n'
    else:
        raise ValueError(data_type)
    with open(reid_filename, 'a') as f:
        line = save_format.format(id=w_reid, t=dt)
        f.write(line)    
 
def write_results(filename, data_type, w_frame_id, w_track_id, dt, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{t},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, t=dt, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)
        #print('save results to {}'.format(filename))

warnings.filterwarnings('ignore')
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color



        
app = QApplication(sys.argv)
#login = LoginScreen()
welcome = WelcomeScreen()
widget = QtWidgets.QStackedWidget()
widget.addWidget(welcome)
widget.setFixedWidth(700)
widget.setFixedHeight(600)
widget.show() 
app.exec()
