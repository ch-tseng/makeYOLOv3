import os

#-------------------------------------------------------------
classes = 12

#Same with you defined in 1_labels_to_yolo_format.py
classList = { "a1":0, "a2":1, "a3":2, "a4":3, "a5":4, "a6":5, "a7":6, "a8":7, "a9": 8, "a10": 9, "a11": 10, "a12": 11 }
folderCharacter = "/"  # \\ is for windows
cfgFolder = "/WORK1/dataset/breads_20191125/cfg.breads_20191125"

#-------------------------------------------------------------

cfg_obj_names = "obj.names"
cfg_obj_data = "obj.data"

if not os.path.exists(cfgFolder + folderCharacter + "weights"):
    os.makedirs(cfgFolder + folderCharacter + "weights")
    print("all weights will generated in here: " + cfgFolder + folderCharacter + "weights" + folderCharacter)


with open(cfgFolder + folderCharacter + cfg_obj_data, 'w') as the_file:
    the_file.write("classes= " + str(classes) + "\n")
    the_file.write("train  = " + cfgFolder + folderCharacter + "train.txt\n")
    the_file.write("valid  = " + cfgFolder + folderCharacter + "test.txt\n")
    the_file.write("names = " + cfgFolder + folderCharacter + "obj.names\n")
    the_file.write("backup = " + cfgFolder + folderCharacter + "weights/")

the_file.close()

print("and cfg folder: " + cfgFolder + " ,is ready for training.")

with open(cfgFolder + folderCharacter + cfg_obj_names, 'w') as the_file:
    for className in classList:
        the_file.write(className + "\n")

the_file.close()

