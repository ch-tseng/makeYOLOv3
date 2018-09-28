import os

classes = 5
classList = ( "chair", "head", "sitting", "standing", "walking" )  #Same with you defined in 1_labels_to_yolo_format.py
folderCharacter = "/"  # \\ is for windows
cfgFolder = "cfg"
cfg_obj_names = "obj.names"
cfg_obj_data = "obj.data"

if not os.path.exists(cfgFolder):
    os.makedirs(cfgFolder)

with open(cfgFolder + folderCharacter + cfg_obj_data, 'w') as the_file:
    the_file.write("classes= " + str(classes) + "\n")
    the_file.write("train  = train.txt\n")
    the_file.write("valid  = test.txt\n")
    the_file.write("names = obj.names\n")
    the_file.write("backup = backup/")

the_file.close()

with open(cfgFolder + folderCharacter + cfg_obj_names, 'w') as the_file:
    for className in classList:
        the_file.write(className + "\n")

the_file.close()
