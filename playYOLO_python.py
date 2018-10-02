from pydarknet import Detector, Image
import imutils
import cv2

video_file = "/media/sf_ShareFolder/videos/p_5.m4v"
output_video = "/media/sf_ShareFolder/pepper_img.avi"

def yoloPython(img):
    img2 = Image(img)
    results = net.detect(img2)

    for cat, score, bounds in results:
        cat = cat.decode("utf-8")
        print(cat)
        if(cat == "0_pepper_flower"):
            boundcolor = (255, 255, 255)
            labelName = "flower"
        elif(cat == "1_pepper_young"):
            boundcolor = (193, 161, 31)
            labelName = "young"
        elif(cat == "2_pepper_matured"):
            boundcolor = (12, 255, 240)
            labelName = "pepper"
        else:
            boundcolor = (255, 255, 255)
            labelName = "unknow"

        x, y, w, h = bounds
        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), 
            int(y + h / 2)), boundcolor, thickness=3)

        boundbox = cv2.imread("cfg.pepper/images/"+cat+".jpg")
        print("boundbox:", boundbox.shape)
        start_x=int(x - w / 2)
        start_y=int(y - h / 2)-boundbox.shape[0]
        '''
        if(start_x<0): start_x = 0
        if(start_y<0): start_y = 0
        end_x=start_x+boundbox.shape[1]
        end_y=start_y+boundbox.shape[0]
        if(end_x>img.shape[1]): end_x = img.shape[1]
        if(end_y>img.shape[0]): end_y = img.shape[0]
        '''
        end_x=start_x+boundbox.shape[1]
        end_y=start_y+boundbox.shape[0]

        print("(end_x-start_x)={}, (end_y-start_y)={}, img.shape[1]={}, img.shape[0]={}".format((end_x-start_x),(end_y-start_y),boundbox.shape[1],boundbox.shape[0]))

        try:
            img[ start_y:end_y, start_x:end_x] = boundbox
            print("read:","images/"+cat+".jpg")

        except:
            print("add text: ",labelName)
            cv2.putText(img, labelName, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1.6, boundcolor, 2)

    return img

net = Detector(bytes("cfg.pepper/yolov3.cfg", encoding="utf-8"),
    bytes("cfg.pepper/weights/yolov3_10000.weights", encoding="utf-8"), 0,
    bytes("cfg.pepper/obj.data",encoding="utf-8"))

cap = cv2.VideoCapture(video_file)
# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_video,fourcc, 30.0, (int(width),int(height)))
i = 0

while True:
    r, img = cap.read()
    img = yoloPython(img)
    out.write(img)
