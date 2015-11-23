import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from PIL import Image

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    print "Cargando datos..."    
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("El set debe ser 'testing' o 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


from pylab import *

images, labels = load_mnist('training')

print "Creando archivos..."
csv = open('etiquetasMNIST.csv','w+')
contador = 0
for img, label in zip(images,labels):
    im = Image.fromarray(img)
    im.save("MNIST/"+str(contador)+'.png')
    etiqueta = str(label).replace("[", "")
    etiqueta = etiqueta.replace("]", "")
    csv.write(etiqueta+",")
    contador = contador+1
print "Archivos creados..."

