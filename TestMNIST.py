import numpy as np
import caffe
import sys
import os

caffe_root = "/home/isaac/caffe/"
MODEL_FILE = "/home/isaac/caffe/examples/mnist/lenet.prototxt"
PRETRAINED = "/home/isaac/caffe/examples/mnist/lenet_iter_100000.caffemodel"
IMAGE_DIR  = "MNIST/"
LABEL_FILE = "lblMNIST.csv"
FILE_FORMAT = ".png"
LeNet = caffe.Classifier( MODEL_FILE, PRETRAINED )

csv = open(LABEL_FILE,"r")
etiquetas = csv.read().split(',')
etiquetas.pop() #eliminar espacio en blanco

rutasIMG = os.listdir(IMAGE_DIR)
#rutasIMG.sort()

correctos = [[0] for x in xrange(10)]
incorrectos = [[0] for x in xrange(10)]

print "\nEJECUTANDO PRUEBA"
print "\nClasificando..."
contador = 0
for etiqueta in etiquetas:
	imagen = IMAGE_DIR+ str(contador)+ FILE_FORMAT
	imagenIN = caffe.io.load_image(imagen, color=False)
	clasificacion = LeNet.predict([imagenIN])	
	res = int(clasificacion[0].argmax())
	#print int(etiqueta) , ',' , imagen , ' - ' , res
	if(int(etiqueta) == res):
		correctos[int(etiqueta)][0] +=1
		#print "\tCorrecto. Contador para ", int(etiqueta), ': ',  correctos[int(etiqueta)][0]
	else:
		incorrectos[int(etiqueta)][0] +=1
		#print "\tIncorrecto"
	contador = contador +1
print "Clasificacion terminada.\n Generando reporte..."


resultado = open("ClasificacionMNIST.csv",'w')
resultado.write("Clase,Clasificado Correcto, Clasificado Erroneo\n")
for numero in xrange(10):
	resultado.write(str(numero) + ',' + str(correctos[numero][0]) + ',' + str(incorrectos[numero][0]) + '\n')
resultado.close()

print "Archivo creado."
