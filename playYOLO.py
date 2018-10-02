import os, time
import argparse
import cv2 as cv
import numpy as np
import webcolors
from colorDetect import DominantColors

#--------------------------------------------------------
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 608       #Width of network's input image
inpHeight = 608      #Height of network's input image

kmean_colors = 5    #detect colors
pad = 30

# Load names of classes
classesFile = "/home/digits/works/makeYOLOv3/cfg.faceYolo/obj.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
 
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "/home/digits/works/makeYOLOv3/cfg.faceYolo/yolov3.cfg";
modelWeights = "/home/digits/works/makeYOLOv3/cfg.faceYolo/weights/yolov3_40000.weights";
 
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

outputFile = FILE_OUTPUT
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

    #actual_name, closest_name = get_colour_name((mean_blue, mean_green, mean_red))

    return actual_name, closest_name, (mean_blue, mean_green, mean_red)
#-----------------------------------------------------------------

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, orgFrame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
 
    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
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
 
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, orgFrame)

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, orgFrame):
    fontSize = 0.9
    fontBold = 2
    center_x = int(left + ((right-left)/2))
    center_y = int(top + ((bottom-top)/2))

    label = '%.2f' % conf
    labelName = '%s:%s' % (classes[classId], label)
    labelColor = (0, 255, 0)

    cv.rectangle(frame, (left, top), (right, bottom), labelColor, 2)
    cv.putText(frame, labelName, (center_x, center_y), cv.FONT_HERSHEY_COMPLEX, fontSize, labelColor, fontBold)

    print(labelName)


args = parser.parse_args()
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo.jpg'

elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo.avi'
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(outputFile ,fourcc, 30.0, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FR$

else:
    # Webcam input
    cap = cv.VideoCapture(0)
 
i = 0
while cv.waitKey(1) < 0:
     
    # get frame from the video
    hasFrame, frame = cap.read()
    i += 1 
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    orgFrame = frame.copy()
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
 
    # Sets the input to the network
    net.setInput(blob)
 
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
 
    # Remove the bounding boxes with low confidence
    postprocess(frame, outs, orgFrame)
 
    # Put efficiency information. The function getPerfProfile returns the 
    # overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

    #cv.putText(frame, label, (30, 40), cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))
 
    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        print("Frame #", i)
        #vid_writer.write(frame.astype(np.uint8))
        out.write(frame)
        cv.imshow("frame", frame)
        cv.waitKey(1)
