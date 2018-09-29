import glob, os
import random
import os.path
import time
from shutil import copyfile
import subprocess
import cv2
from xml.dom import minidom
from os.path import basename
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#--------------------------------------------------------------------
folderCharacter = "/"  # \\ is for windows
xmlFolder = "../datasets/cucumber_A/labels"
imgFolder = "../datasets/cucumber_A/images"
saveYoloPath = "../datasets/cucumber_A/yolo"
classList = { "0_cucumber_flower":0, "2_cucumber_matured": 1 }

modelYOLO = "yolov3"  #yolov3 or yolov3-tiny
testRatio = 0.2
cfgFolder = "cfg.cucumber_A"
cfg_obj_names = "obj.names"
cfg_obj_data = "obj.data"

numBatch = 24
numSubdivision = 8
darknetEcec = "../darknet/darknet"

#---------------------------------------------------------------------

if not os.path.exists(saveYoloPath):
    os.makedirs(saveYoloPath)

def downloadPretrained(url):
    import wget
    print("Downloading the pretrained model darknet53.conv.74, please wait.)
    wget.download(url)

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

        if(os.path.isfile(xmlfile)):
            print("id:{}".format(fileCount))
            print("processing {}".format(imgfile))
            print("processing {}".format(xmlfile))
            fileCount += 1

            transferYolo( xmlfile, imgfile, "")
            copyfile(imgfile, saveYoloPath + folderCharacter + file)

# step2 ---------------------------------------------------------------
fileList = []
outputTrainFile = cfgFolder + "/train.txt"
outputTestFile = cfgFolder + "/test.txt"

if not os.path.exists(cfgFolder):
    os.makedirs(cfgFolder)

for file in os.listdir(saveYoloPath):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".b$
        fileList.append(saveYoloPath + folderCharacter + file)

print("total image files: ", len(fileList))

testCount = int(len(fileList) * testRatio)
trainCount = len(fileList) - testCount

a = range(len(fileList))
test_data = random.sample(a, testCount)
train_data = random.sample(a, trainCount)

print ("Train:{} images".format(len(train_data)))
print("Test:{} images".format(len(test_data)))

with open(outputTrainFile, 'a') as the_file:
    for i in train_data:
        the_file.write(fileList[i] + "\n")

the_file.close()

with open(outputTestFile, 'a') as the_file:
    for i in test_data:
        the_file.write(fileList[i] + "\n")

the_file.close()

# step2 -------------------------------------------

classes = len(classList)

if not os.path.exists(cfgFolder + folderCharacter + "weights"):
    os.makedirs(cfgFolder + folderCharacter + "weights")
    print("all weights will generated in here: " + cfgFolder + folderCharacter + "weights" + folderCharacte$


with open(cfgFolder + folderCharacter + cfg_obj_data, 'w') as the_file:
    the_file.write("classes= " + str(classes) + "\n")
    the_file.write("train  = " + cfgFolder + folderCharacter + "train.txt\n")
    the_file.write("valid  = " + cfgFolder + folderCharacter + "test.txt\n")
    the_file.write("names = " + cfgFolder + folderCharacter + "obj.names\n")
    the_file.write("backup = " + cfgFolder + folderCharacter + "weights/")

the_file.close()

print("    cfg folder: " + cfgFolder + " ,is ready for training.")

with open(cfgFolder + folderCharacter + cfg_obj_names, 'w') as the_file:
    for className in classList:
        the_file.write(className + "\n")

the_file.close()

# step4 ----------------------------------------------------

if not os.path.exists("darknet53.conv.74"):
    downloadPretrained("https://pjreddie.com/media/files/darknet53.conv.74")

classNum = len(classList)
filerNum = (classNum + 5) * 3

if(modelYOLO == "yolov3"):
    fileCFG = "yolov3.cfg"

else:
    fileCFG = "yolov3-tiny.cfg"

with open("cfg"+folderCharacter+fileCFG) as file:
    file_content = file.read()

file.close

file_updated = file_content.replace("{BATCH}", numBatch)
file_updated = file_updated.replace("{SUBDIVISIONS}", numSubdivision)
file_updated = file_updated.replace("{FILTERS}", filterNum)
file_updated = file_updated.replace("{CLASSES}", classNum)

file = open(cfgFolder+folderCharacter+fileCFG, "w")
file.write(file_updated)
file.close

executeCmd = darknetEcec + " detector train " + cfgFolder + folderCharacter + 
    "obj.data " + cfgFolder + folderCharacter + fileCFG + " darknet53.conv.74"

print("Execute darknet training command:")
print("    " + executeCmd)
print("    You can find weights files here:" + cfgFolder + folderCharacter + "weights" + folderCharacter)

subprocess.Popen(executeCmd)

