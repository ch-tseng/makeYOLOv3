import random
import glob, os
import os.path

#---------------------------------------------------------
testRatio = 0.2
imageFolder = "../datasets/cucumber_A/yolo"
cfgFolder = "cfg.cucumber_A"
folderCharacter = "/"  # \\ is for windows
#--------------------------------------------------------

fileList = []
outputTrainFile = cfgFolder + "/train.txt"
outputTestFile = cfgFolder + "/test.txt"

if not os.path.exists(cfgFolder):
    os.makedirs(cfgFolder)

for file in os.listdir(imageFolder):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        fileList.append(imageFolder + folderCharacter + file)

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

