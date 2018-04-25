import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#bianry imge ming aread Rectangle without rataion
class binaryMinAreaRect(object):
    def __init__(self,image_data):
        self.image=np.asarray(image_data).astype(np.uint8)
        self.box=[0]*4
    def get_box(self):
        adjust=1
        Height,Width=self.image.shape
        for w in range(Width):
            if (np.sum(self.image[:, w]) != Height):
                if(w>0):
                    w=w-adjust
                else:
                    pass
                self.box[0] = w  # left,minus 1 piexl
                break;
            else:
                continue
        for h in range(Height):
            if (np.sum(self.image[h, :]) != Width):
                if(h>0):
                    h=h-adjust
                else:
                    pass
                self.box[1] = h  # top minus 1 piexl
                break;
            else:
                continue
        for w in range(Width - 1, -1, -1):
            if (np.sum(self.image[:, w]) != Height):
                if (w < Width-adjust):
                    w = w + adjust
                else:
                    pass
                self.box[2] = w+adjust  # right plus 1 piexl
                break;
            else:
                continue
        for h in range(Height - 1, -1, -1):
            if (np.sum(self.image[h, :]) != Width):
                if (h <= Height-adjust):
                    h = h + adjust
                else:
                    pass
                self.box[3] = h # bottom plus 2 piexl
                break;
            else:
                continue
        return self.box