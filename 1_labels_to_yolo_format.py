import glob, os
import os.path
import time
import cv2
from xml.dom import minidom
from os.path import basename
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

folderCharacter = "/"  # \\ is for windows
xmlFolder = "misoffice-actions/labels"
imgFolder = "misoffice-actions/images"
saveYoloPath = "misoffice-actions/yolo"
classList = { "chair":0, "head": 1, "sitting":2, "standing": 3, "walking": 4 }

if not os.path.exists(saveYoloPath):
    os.makedirs(saveYoloPath)


def transferYolo( xmlFilepath, imgFilepath, labelGrep=""):
    global imgFolder
    
    img_file, img_file_extension = os.path.splitext(imgFilepath)
    img_filename = basename(img_file)
    print(imgFilepath)
    img = cv2.imread(imgFilepath)
    imgShape = img.shape
    print (img.shape)
    img_h = imgShape[0]
    img_w = imgShape[1]

    labelXML = minidom.parse(xmlFilepath)
    labelName = []
    labelXmin = []
    labelYmin = []
    labelXmax = []
    labelYmax = []
    totalW = 0
    totalH = 0
    countLabels = 0

    tmpArrays = labelXML.getElementsByTagName("filename")
    for elem in tmpArrays:
        filenameImage = elem.firstChild.data

    tmpArrays = labelXML.getElementsByTagName("name")
    for elem in tmpArrays:
        labelName.append(str(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmin")
    for elem in tmpArrays:
        labelXmin.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymin")
    for elem in tmpArrays:
        labelYmin.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmax")
    for elem in tmpArrays:
        labelXmax.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymax")
    for elem in tmpArrays:
        labelYmax.append(int(elem.firstChild.data))

    yoloFilename = saveYoloPath + folderCharacter + img_filename + ".txt"
    print("writeing to {}".format(yoloFilename))

    with open(yoloFilename, 'a') as the_file:
        i = 0
        for className in labelName:
            if(className==labelGrep or labelGrep==""):
                classID = classList[className]
                x = (labelXmin[i] + (labelXmax[i]-labelXmin[i])/2) * 1.0 / img_w 
                y = (labelYmin[i] + (labelYmax[i]-labelYmin[i])/2) * 1.0 / img_h
                w = (labelXmax[i]-labelXmin[i]) * 1.0 / img_w
                h = (labelYmax[i]-labelYmin[i]) * 1.0 / img_h

                the_file.write(str(classID) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
                i += 1

    the_file.close()

#---------------------------------------------------------------
fileCount = 0

for file in os.listdir(imgFolder):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
        imgfile = imgFolder + folderCharacter + file
        xmlfile = xmlFolder + folderCharacter + filename + ".xml"

        print("id:{}".format(fileCount))
        print("processing {}".format(imgfile))
        print("processing {}".format(xmlfile))
        fileCount += 1

        transferYolo( xmlfile, imgfile, "")

