import numpy as np
from PIL import Image

img_dat = []

with open("./15x224_1-1.png.dat", "r") as f:

    data = f.read()

    splt = data.split(" ") 

    for x in splt:

        if x == '':
            continue 

        img_dat.append(float(x) * 1000)


arr = np.reshape(np.array(img_dat), (int(240 / 16), int(3584 / 16)))
Image.fromarray(arr).convert("L").show()
