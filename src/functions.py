import os
import  numpy as np
import cv2
from .binaryMinAreaRect import binaryMinAreaRect
from PIL import Image
import copy
from .graham import convex_hull_graham
import math

###############################GBDT feature extract###################################
##
##
##################################################################################
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
##############################################################################################

class Past2panel(object):
    def __init__(self,imge_data,shape=(256,256,3)):
        self.img = imge_data
        self.src_shape=imge_data.shape
        self.channels =1 if len(self.src_shape)>2 else 0
        self.shape=shape

    def upresize(self):
        #flag to determine if the target shape is large than orignal shape
        flag=0
        if(self.channels):
            imge_h, imge_w, channels = self.img.shape

        else:
            imge_h, imge_w = self.img.shape
            channels=1

        rest = np.ones(self.shape, dtype=np.uint8)*255
        imgh_half = imge_h // 2
        imgw_half = imge_w // 2
        if(channels>1):
            for c in range(channels):
                for row in range(imge_h):
                    for col in range(imge_w):
                        rest[row + self.shape[0] // 2 - imgh_half, \
                                col + self.shape[1] // 2 - imgw_half,c] = self.img[row, col, c]
        else:
            for c in range(channels):
                for row in range(imge_h):
                    for col in range(imge_w):
                        rest[row + self.shape[0] // 2 - imgh_half,\
                                col + self.shape[1] // 2 - imgw_half] = self.img[row, col]

        return rest
#adjust imges width and height
class adjust_wh(object):
    def __init__(self,img_src,shape=(256,256,3)):
        self.src_img=img_src
        self.src_hw=img_src.shape[:2]
        self.shape=shape
    def adjust(self):
        # print(self.src_img.shape)
        img_tmp=copy.deepcopy(self.src_img)
        src_box=binaryMinAreaRect(img_tmp>140).get_box()
        img=Image.fromarray(img_tmp)
        src_img=np.asarray(img.crop(src_box))

        # del img_tmp
        return Past2panel(src_img, self.shape).upresize()

def fetch_file_name(file_path,fmt):
    Names = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1] == fmt:
                Names.append(os.path.join(root, file))
    return Names
def mkdir(path):
    path = path.strip()
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        print(path + ' 创建成功')
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False