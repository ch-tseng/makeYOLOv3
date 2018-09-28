import os

#-------------------------------------------------------------
classes = 2

#Same with you defined in 1_labels_to_yolo_format.py
classList = { "0_cucumber_flower":0, "2_cucumber_matured": 1 }

folderCharacter = "/"  # \\ is for windows
cfgFolder = "cfg.cucumber_A"

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

