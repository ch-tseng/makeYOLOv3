import os, time
import argparse
import cv2
import numpy as np

#--------------------------------------------------------
modelType = "yolo"  #yolo or yolo-tiny
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold

classesFile = "../darknet/data/coco.names";
modelConfiguration = "../darknet/cfg/yolov3.cfg";
modelWeights = "../darknet/weights/yolov3.weights";

displayScreen = False  #Do you want to show the image on LCD?
outputToFile = True   #output the predicted result to image or video file

#Label & Box
fontSize = 0.35
fontBold = 1
labelColor = (0,0,255)
boxbold = 1
boxColor = (255,255,255)
#--------------------------------------------------------

if(modelType=="yolo"):
    inpWidth = 608       #Width of network's input image
    inpHeight = 608      #Height of network's input image
else:
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image


classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
 
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

parser = argparse.ArgumentParser(description="Do you wish to scan for live hosts or conduct a port scan?")
parser.add_argument("-i", dest='image', action='store', help='Image')
parser.add_argument("-v", dest='video', action='store',help='Video file')

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def getROI_Color(roi):
    mean_blue = np.mean(roi[:,:,0])
    mean_green = np.mean(roi[:,:,1])
    mean_red = np.mean(roi[:,:,2])
    actual_name, closest_name = get_colour_name((mean_red, mean_green, mean_blue))

    return actual_name, closest_name, (mean_blue, mean_green, mean_red)
#-----------------------------------------------------------------

# Get the names of the output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()

    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def postprocess(frame, outs, orgFrame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
 
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
 
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, orgFrame)

def drawPred(classId, conf, left, top, right, bottom, orgFrame):
    label = '%.2f' % conf
    labelName = '%s:%s' % (classes[classId], label)

    cv2.rectangle(frame, (left, top), (right, bottom), boxColor, boxbold)
    cv2.putText(frame, labelName, (left, top-10), cv2.FONT_HERSHEY_COMPLEX, fontSize, labelColor, fontBold)

    print(labelName)


args = parser.parse_args()
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo.jpg'

elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(outputFile, fourcc, 30.0, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

else:
    # Webcam input
    cap = cv2.VideoCapture(0)

i = 0
while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()

    i += 1 
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv2.waitKey(3000)
        break

    orgFrame = frame.copy()

    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs, orgFrame)

    t, _ = net.getPerfProfile()
    #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

    if (args.image):

        if(outputToFile):
            cv2.imwrite(outputFile, frame.astype(np.uint8))

        if(displayScreen):
            cv2.imshow("Predicted", frame)

    else:
        print("Frame #{} processed.".format(i))

        if(outputToFile):
            out.write(frame)

        if(displayScreen):
            cv.imshow("frame", frame)
            cv.waitKey(1)
