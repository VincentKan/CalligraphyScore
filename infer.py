import pickle
from PIL import Image
import numpy as np
from CalligraphyCmp.src.functions import fetch_file_name,mkdir,get_features
modelname="finalized_model.sav"
clf = pickle.load(open(modelname, 'rb'))
img_path="/home/william/Master/Font_sim/CalligraphyCmp/synthesis/img/tj/"
Names=fetch_file_name(img_path,".bmp")
i=0
numbers=len(Names)
X_infer = np.empty((numbers, 8))
for file_name in Names:
    img = Image.open(file_name).convert("RGB")
    img = np.asarray(img)
    features=get_features(img)

    X_infer[i] = np.asarray(features[0:], dtype=np.float64)
    i+=1

result=clf.predict(X_infer)
print(np.sum(result)/numbers)