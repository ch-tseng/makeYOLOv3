import os, time
import subprocess

cfgFolder = "cfg.cucumber_A"
modelYOLO = "yolov3"  #yolov3 or yolov3-tiny
classList = { "0_cucumber_flower":0, "2_cucumber_matured": 1 }
folderCharacter = "/"  # \\ is for windows

numBatch = 24
numSubdivision = 8
darknetEcec = "../darknet/darknet"
#------------------------------------------------------

def downloadPretrained(url):
    import wget
    print("Downloading the pretrained model darknet53.conv.74, please wait.)
    wget.download(url)

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

subprocess.Popen(executeCmd)
