import cv2
import numpy as np
import matplotlib.pyplot as plt
from .src.graham import convex_hull_graham
from .src.functions import fetch_file_name,mkdir
import math
import os
import csv
def HistStandD(img,binaryThresh=127):
    if len(img.shape) >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape
    x_P2y = np.arange(0, W, 1)
    y_P2y = np.sum(img[:, i] < binaryThresh for i in x_P2y)
    # del x_src,x_tar
    x_P2x = np.arange(0, H, 1)
    y_P2x = np.sum(img[i, :] < binaryThresh for i in x_P2x)

    return np.std(y_P2x)/np.std(y_P2y)

def get_features(imgdata,thresh=150):
    imggray=cv2.cvtColor(imgdata,cv2.COLOR_BGR2GRAY)
    ret,binary=cv2.threshold(imggray,thresh,255,0)
    _,contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>1:
        contours = np.vstack(contours[1:]).squeeze()
    lst=[]
    for item in contours:
        lst.append((item[0],item[1]))
    # print(lst)
    convex=np.asarray(convex_hull_graham(lst))
    # print(convex)
    ConvextArea = cv2.contourArea(convex)
    # print("ConvextArea", ConvextArea)
    #Contour Perimeter
    ConvextPerimeter = cv2.arcLength(convex,True)
    # print("ConvextPerimeter",ConvextPerimeter)
    ContourArea = cv2.contourArea(contours)
    # print("ContourArea", ContourArea)
    ContourPerimeter=cv2.arcLength(contours,True)
    # print("ContourPerimeter", ContourPerimeter)
    # Aspect Ratio
    # It is the ratio of width to height of bounding rect of the object#
    x,y,w,h = cv2.boundingRect(contours)
    aspect_ratio = float(w) / h
    # print("Aspect ratio", aspect_ratio)
    BRimg = cv2.rectangle(imgdata, (x, y), (x + w, y + h), (0, 255, 0), 2)
    rect_area = w*h
    #Extent is the ratio of contour area to bounding rectangle area.
    extent = float(ContourArea)/rect_area
    # print("Extent",extent)
    #
    # Convecity is the ratio of contour area to its convex hull area.
    Convecity = float(ContourArea)/ConvextArea
    Rectangularity=ConvextPerimeter/(2*w+2*h)
    # print("Convecity",Convecity)
    # print("Rectangularity", Rectangularity)
    #gravity
    BRContours=[(x,y),(x+w,h),(x+w,y+h),(x,y+h),(x,y)]
    BRContours=np.asarray(BRContours)
    # BRContours=BRContours.extend(BRContours[i] for i in range(1, len(BRContours) - 1))
    M = cv2.moments(BRContours)
    C=0
    X_T=0
    Y_T=0
    for i in range(x,x+w):
        for j in range(y,y+h):
            if binary[i,j]==0:
                X_T += i
                Y_T +=j
                C+=1

    cx=int(X_T/C)
    cy=int(Y_T/C)
    # print("Gravity", (cx, cy))
    # cx = int(M['m10']/M['m00'])
    # cy = int(M['m01']/M['m00'])
    RPA=(ContourArea-ContourPerimeter/4)/(ContourArea-math.sqrt(ContourArea))
    # print("RPA",RPA)
    f1=round(HistStandD(imgdata,thresh),5)
    f2=round(cx/w,5)
    f3=round(cy/h,5)
    f4=round(aspect_ratio,5)
    f5=round(extent,5)
    f6=round(RPA,5)
    f7=round(Convecity,5)
    f8=round(Rectangularity,5)
    # plt.plot(convex[:, 0], convex[:, 1], 'b-', picker=5)  # Plot lines
    # plt.plot(cx, cy, "r^", picker=5)
    # plt.imshow(BRimg)
    # plt.show()
    return [f1,f2,f3,f4,f5,f6,f7,f8]

def writeF2csv(file_names,outputpath,needscore=False,score=100):
    fetures_list=[]
    savename=os.path.join(outputPath,'data.csv')
    if needscore:
        for fileName in file_names:
            char_name=os.path.basename(fileName)
            pardir=fileName.split('/')[-2]
            img=cv2.imread(fileName)
            features=get_features(img)
            features.insert(0,pardir+char_name)
            features.append(score)
            # print(fileName)
            # print(features)
            fetures_list.append(features)
    else:
        for fileName in file_names:
            char_name=os.path.basename(fileName)
            pardir=fileName.split('/')[-2]
            img=cv2.imread(fileName)
            features=get_features(img)
            features.insert(pardir+char_name,0)
            # print(fileName)
            # print(features)
            fetures_list.append(features)

    mkdir(outputpath)
    try:
        with open(savename,'a',newline='') as csvfile:
            filewriter=csv.writer(csvfile)
        # file=open(savename,'wa')
            for items in fetures_list:
                filewriter.writerow(items)
            # for item in items:
            #     file.write(item)
            #     file.write(',')
            # file.write("\n")
    except Exception:
        print("数据写入失败")
    finally:
        csvfile.close()
        del fetures_list


if __name__=="__main__":
    ink_path="/home/william/Master/Font_sim/CalligraphyCmp/synthesis/Tml/add/hao/"
    # cht_path=""
    # syn_path=""
    outputPath="/home/william/Master/Font_sim/CalligraphyCmp/synthesis/Tml/"
    file_names=fetch_file_name(ink_path,".bmp")
    print(file_names)
    # for filename in file_names:
    #     img = Image.open(filename).convert("RGB")
    #     img = np.asarray(img)
    #     misc.imsave(filename, img)
    # cht_names=fetch_file_name(cht_path,".bmp")
    writeF2csv(file_names,outputPath,True,85)


