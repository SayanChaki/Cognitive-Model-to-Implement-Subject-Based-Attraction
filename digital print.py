import cv2
from matplotlib import pyplot as plt
import dlib
from scipy.spatial import distance
import os
import pandas as pd
import numpy as np
from PIL import Image
import dlib

Row2=[]
images = []
i='0'
for filename in os.listdir('test_pics'):
    img = cv2.imread(os.path.join('test_pics',filename))
    
    i=str(int(i)+1)

    plt.figure('Face'+i)
    plt.imshow(img)
    plt.show()

    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plt.figure('FaceGray'+i)
    plt.imshow(gray)
    plt.show()
    
    
    
    faces=detector(gray)
    
    for face in faces:
        x1=face.left()
        x2=face.right()
        y1=face.top()
        y2=face.bottom()
    
    face_points=[]
    landmarks=predictor(image=gray, box=face)

    for n in range(0,68):
        x=landmarks.part(n).x
        y=landmarks.part(n).y
        cv2.circle(img=img, center=(x,y), radius=5, color=(0,255,0), thickness=-1)
        face_points.append((x,y))
        print(n,(x,y))
    
    data2=['Subject'+i]
    LBL=face_points[0]
    RBL=face_points[16]
    dBl=distance.euclidean(LBL,RBL)
    data2.append(dBl)

    Nose_up=face_points[27]
    Nose_tip=face_points[33]
    d_Noselength=distance.euclidean(Nose_up, Nose_tip)
    data2.append(d_Noselength)

    Nose_width_left=face_points[31]
    Nose_width_right=face_points[35]
    d_Nose_width=distance.euclidean(Nose_width_left, Nose_width_right)
    data2.append(d_Nose_width)
    
    Chin_width_left=face_points[7]
    Chin_width_right=face_points[9]
    d_Chin_width=distance.euclidean(Chin_width_left,Chin_width_right)
    data2.append(d_Chin_width)
    
    Lip_width_left=face_points[48]
    Lip_width_right=face_points[54]
    d_Lip_width=distance.euclidean(Lip_width_left,Lip_width_right)
    data2.append(d_Lip_width)
    
    Lip_Height_upper_up=face_points[51]
    Lip_Height_upper_down=face_points[62]
    d_lip_Height_upper=distance.euclidean(Lip_Height_upper_up,Lip_Height_upper_down)
    data2.append(d_lip_Height_upper)
    
    Lip_Height_lower_up=face_points[66]
    Lip_Height_lower_down=face_points[57]
    d_Lip_Height_lower=distance.euclidean(Lip_Height_lower_up,Lip_Height_lower_down)
    data2.append(d_Lip_Height_lower)
    
    Eye_width_left=face_points[36]
    Eye_width_right=face_points[39]
    d_Eye_width=distance.euclidean(Eye_width_left,Eye_width_right)
    data2.append(d_Eye_width)
    
    Eye_height_up=face_points[38]
    Eye_height_lower=face_points[40]
    d_Eye_height=distance.euclidean(Eye_height_lower,Eye_height_up)
    data2.append(d_Eye_height)
    
    LEye=face_points[39]
    REye=face_points[42]
    d_Eye=distance.euclidean(LEye,REye)
    data2.append((d_Eye ))
    
    jawline_tangent_base=distance.euclidean(face_points[66],face_points[4])
    jawline_tangent_perpendicular=distance.euclidean(face_points[66],face_points[8])
    tangent_jawline=float(jawline_tangent_perpendicular/jawline_tangent_base)
    
    face_tangent=distance.euclidean(face_points[27], face_points[8])/distance.euclidean(face_points[27],face_points[0])
   
    (x1,y1)=(face_points[36])
    (x3,y3)=face_points[39]
    (x4,y4)=((x1+x3)/2,(y1+y3)/2)
    (x2,y2)=face_points[20]
    
    tangent_forehead=(x2-x4)/(y2-y4)
    
    data2.append(float(dBl/d_Chin_width))
    data2.append(float(d_Noselength/d_Nose_width))
    data2.append(float(d_Lip_Height_lower/d_Lip_width))
    data2.append(float(d_lip_Height_upper/d_Lip_width))
    data2.append(float(d_Eye/dBl))
    data2.append(float(d_Eye_width/d_Eye_height))
    data2.append(tangent_jawline)
    data2.append(face_tangent)
    data2.append(tangent_forehead)
    
    
    
    Y=input("Are yu attracted?")
    data2.append(Y)
    Row2.append(data2)
    print(data2)

    
    
    plt.figure('Face'+i)
    plt.imshow(img)
    plt.show()


print(Row2)

Attraction4=pd.DataFrame(Row2,columns=['Name','Broad_Length','Nose Length','Nose Width','Chin Width','Lip Width','Upper Lip Height',
                                       'Lower Lip Height','Eye Width','Eye Height','Eye Distance','Face Ratio','Nose Ratio','Lower Lip Ratio','Upper Lip Ratio','Eye-face ratio','Eye Ratio','Jawline Tangent','Face Tangent','Forehead Tangent','Attracted'])
Attraction4.to_csv('Attraction5 _training.csv') 

    
    
    
    
        