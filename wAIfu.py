import requests
import json
import shutil
import sys
import os
from PIL import Image
from PyQt6.QtWidgets import QApplication, QSizePolicy, QMainWindow, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QComboBox, QFileDialog, QSpacerItem, QMessageBox
from PyQt6.QtGui import QPixmap, QDesktopServices, QFontMetrics
from PyQt6.QtCore import Qt, QUrl, QTimer, QFileInfo
import random
import time
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import webbrowser
import subprocess
import scipy
import winshell
from win32com.client import Dispatch
import glob

version = "NSFW" # NSFW or SFW - also if i wanted to make a version without access to NSFW what-so-ever

NSFW = False
predictedFolder = 0
currentFile = ""
image_data = list()
image_name = "" # Signature
ext = "" # extension
bg_color = "" # dominant color
source = "" # source
chibiFolder = "Assets"
chibi_path = "Assets/idle.png"
totalRated = 0
currentPick = 0
chibiExists = False
modelExists = False
trainingData = []
defaultTrainingData = ""
trainingPredicted = 0

if glob.glob("*.h5"):
    for file_path in glob.glob("*.h5"):
        trainingData.append(file_path)
    print(trainingData)
    defaultTrainingData = trainingData[len(trainingData)-1]
    print(defaultTrainingData)


# Create a list of names
filename = "names.txt"
os.makedirs("images", exist_ok=True) # Create folder if it doesn't exist
os.makedirs("ungraded", exist_ok=True) # Create folder if it doesn't exist
os.makedirs("test", exist_ok=True) # Create folder if it doesn't exist
os.makedirs("validation", exist_ok=True) # Create folder if it doesn't exist
for x in range(1,11):
    os.makedirs(os.path.join("validation",str(x)), exist_ok=True) # Create folder if it doesn't exist
    os.makedirs(os.path.join("images",str(x)), exist_ok=True) # Create folder if it doesn't exist

# Wipe ungraded folder
files = os.listdir("ungraded")

for file_name in files:
    file_path = os.path.join("ungraded", file_name)
    os.remove(file_path)

# Wipe ungraded folder
files2 = os.listdir("test")

for file_name in files2:
    file_path2 = os.path.join("test", file_name)
    os.remove(file_path2)

def chibiChangeImage(photoPick):
    global chibi_path

    global timerFrame
    if timerFrame == "point":
        changeTo = 1
        timer.start(10*1000)
        timerFrame = "s1"
    elif timerFrame == "s1":
        changeTo = 4
        timer.start(500)
        timerFrame = "s2"
    elif timerFrame == "s2":
        changeTo = 5
        timer.start(500)
        timerFrame = "s3"
    elif timerFrame == "s3":
        changeTo = 6
        timer.start(500)
        timerFrame = "s4"
    elif timerFrame == "s4":
        changeTo = 7
        timer.start(500)
        timerFrame = "s1"
    elif timerFrame == "none":
        if photoPick == -1:
            changeTo = 4
        elif photoPick <= 3:
            changeTo = 3
        elif photoPick < 7:
            changeTo = 0
        else:
            changeTo = 2
    match (changeTo):
        case 0:
            chibi_path = str(chibiFolder) + "/idle.png"
            timerFrame = "point" # remember this
            timer.start(5*1000) # (1000ms = 1s)
        
        case 1:
            chibi_path = str(chibiFolder) + "/point.png"

        case 2:
            chibi_path = str(chibiFolder) + "/excited.png"
            timerFrame = "point" # remember this
            timer.start(5*1000) # (1000ms = 1s)

        case 3:
            chibi_path = str(chibiFolder) + "/shock.png"
            timerFrame = "point" # remember this
            timer.start(5*1000) # (1000ms = 1s)
        
        case 4:
            chibi_path = str(chibiFolder) + "/sleepframes/1.png"

        case 5:
            chibi_path = str(chibiFolder) + "/sleepframes/2.png"

        case 6:
            chibi_path = str(chibiFolder) + "/sleepframes/3.png"

        case 7:
            chibi_path = str(chibiFolder) + "/sleepframes/4.png"
            
    global chibi
    chibi_pixmap = QPixmap(chibi_path)
    maxWidth = int(968/2.65)
    maxHeight = int(1450/2.65)
    chibi_pixmap = chibi_pixmap.scaled(maxWidth, maxHeight, Qt.AspectRatioMode.KeepAspectRatio)
    chibi.setPixmap(chibi_pixmap)


# Functions:
def use_model(name):
    # Define the directories where the images are stored
    test_dir = 'test/'
    os.makedirs(test_dir, exist_ok=True)
    image = Image.open(os.path.join("ungraded",currentFile))
    image.thumbnail((178,218), Image.LANCZOS)
    image.save(os.path.join(test_dir,currentFile))
    

    # Load the saved model
    model = load_model(name)

    # Define the image size
    img_size = (178, 218)

    # Define the label names
    label_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    # Loop through each image in the test directory and classify it
    for file in os.listdir(test_dir):
        # Load the image and convert to RGB mode
        img = Image.open(os.path.join(test_dir, file)).convert('RGB')

        # Resize the image
        img = img.resize(img_size)

        # Convert the image to a numpy array
        img_array = np.array(img, dtype=np.float32)

        # Reshape the image to a single batch with one channel
        img_batch = np.expand_dims(img_array, axis=0)

        # Normalize the image
        img_batch /= 255.

        # Fix the input shape
        img_batch = np.transpose(img_batch, (0, 2, 1, 3))

        # Predict the class label
        prediction = model.predict(img_batch)

        # Get the index of the highest probability
        predicted_class = np.argmax(prediction[0])

        # Print the predicted label
        print("File:", file, "Predicted Label:", label_names[predicted_class],"out of 10 using:",name)
        global predictedFolder
        predictedFolder = label_names[predicted_class]
        if chibiExists:
            global timerFrame
            timerFrame = "none"
            chibiChangeImage(int(predicted_class))
        global trainingPredicted
        trainingPredicted = (label_names[predicted_class])
        os.remove(os.path.join(test_dir,currentFile))
        
def trainClicked():
    train.setText("Training..")
    for i in range(10):
        buttons[i].setVisible(False)
    popup = QMessageBox()
    popup.setWindowTitle("Training..")
    popup.setText("The AI will now begin training. This may take a few minutes.\nPlease place the file in the wAIfu directory. You can name it whatever.\nYou will get a popup message once it has trained.")
    popup.exec()
    window.update()
    train_model()
    trainingRefresh()

def train_model():
    file_dialog = QFileDialog()
    file_dialog.setDefaultSuffix(".h5")  # Set the default file extension to .txt
    file_dialog.setNameFilters(["TrainingData (*.h5)"])  # Set the file filters
    file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)  # Set the dialog mode to save a file
    if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
        file_path = file_dialog.selectedFiles()[0]

        # Define the directories where the images are stored
        train_dir = 'images'
        validation_dir = 'validation'

        # Set up the image data generator
        train_datagen = ImageDataGenerator(rescale=1./255)
        validation_datagen = ImageDataGenerator(rescale=1./255)

        # Set up the batch sizes
        batch_size = 32

        # Define the number of classes (in this case, 10)
        num_classes = 10

        # Set up the training data generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(178,218),
            batch_size=batch_size,
            class_mode='categorical')

        # Set up the validation data generator
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(178,218),
            batch_size=batch_size,
            class_mode='categorical')

        # Create the model - this isn't the most efficient, but it'll do for my first solo AI project
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(178,218, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=0.001),
                    metrics=['accuracy'])

        # Train the model
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=10,
            validation_data=validation_generator,
            validation_steps=len(validation_generator))

        # Saving
        model.save(str(file_path))
        train.setText("Finished")
        #use_model(str(os.path.basename(file_path)))
        popup = QMessageBox()
        popup.setWindowTitle("Training Finished")
        popup.setText(f"The AI has finished training and saved your file at:\n{file_path}")
        popup.exec()
        window.update()
    else:  
        train.setText("Cancelled")
        print("Training ejected.")
    for i in range(10):
        buttons[i].setVisible(True)


def augmentData(path,output,numb,amount):
    augmentedImages = []
    fileoutput = []

    # Open the image at the given path
    with Image.open(path) as img:
        img.thumbnail((178,218), Image.LANCZOS)
        img = img.convert('RGB')
        r, g, b = img.split()
        methods = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90,
        Image.ROTATE_180, Image.ROTATE_270, Image.TRANSPOSE, Image.TRANSVERSE]
        for x in range(0,amount): #10 long
            method = random.choice(methods)
            augmentedImages.append(img.transpose(method))
            if x > amount/2:
                r, g, b = img.split()
                colors = [r,g,b]
                channels = []
                for z in range(0,3):
                    channels.append(colors[random.randint(0,2)])
                random.shuffle(channels)
                augmentedImages[x] = Image.merge("RGB", channels)
            # Get the extension
            ext = os.path.splitext(path)[1]

            # Construct new file names based on the original file name
            file_base = output + os.path.join("\\",numb) + currentFile.replace(ext,"")
            fileoutput.append(file_base + "_edited_" + str(x) + ext)
            augmentedImages[x].save(fileoutput[x])


def check_name(name):
    if os.path.exists(filename) == False:
        open(filename,"x")

    with open(filename, "r") as file:
        contents = file.read()
        if name in contents: 
            print(f"Duplicate: {name} is in the file.")
            return False
        else:
            return True

def getImage(amount):
    os.makedirs("ungraded", exist_ok=True)
    for x in range(1,amount+1):
        if version == "NSFW":
            if NSFW:
                response = requests.get("https://api.waifu.im/search/?included_tags=hentai")
            else:
                response = requests.get("https://api.waifu.im/search/?included_tags=waifu")
        else:
            response = requests.get("https://api.waifu.im/search/?included_tags=waifu")

        json_data = json.loads(response.text)
        # Open images dic
        image_data = []
        for item in json_data["images"]:
            for key, value in item.items():
                # Remove unwanted characters
                image_data.append(key.translate(str.maketrans("", "", '{[}]:,"')))
                # If the value is a dictionary
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        # Remove unwanted characters
                        image_data.append(sub_key.translate(str.maketrans("", "", '{[}]:,"')))
                        image_data.append(str(sub_value).translate(str.maketrans("", "", '{[}]:,"')))
                elif isinstance(value, list):
                    for sub_value in value:
                        # If an element is a dictionary, iterate over its keys and image_data
                        if isinstance(sub_value, dict):
                            for sub_key, sub_sub_value in sub_value.items():
                                # Remove unwanted characters
                                image_data.append(sub_key.translate(str.maketrans("", "", '{[}]:,"')))
                                image_data.append(str(sub_sub_value).translate(str.maketrans("", "", '{[}]:,"')))
                        # Otherwise, append the element to the image_data list
                        else:
                            image_data.append(str(sub_value).translate(str.maketrans("", "", '{[}]:,"')))
                # Otherwise, append the value to the image_data list
                else:
                    image_data.append(str(value).translate(str.maketrans("", "", '{[}]:,"')))
        url = image_data[image_data.index('url')+1]
        url = url.replace("https//","https://")
        global source
        source = image_data[image_data.index('source')+1]
        source = source.replace("https//","https://")
        global ext
        ext = image_data[image_data.index('extension')+1]
        image_name = image_data[image_data.index('signature')+1]+ext
        global bg_color
        bg_color = image_data[image_data.index('dominant_color')+1]
        if ext == ".gif":
            with open(filename, "a") as file:
                file.write(image_name + "\n")
            print("Cannot show gif.")
            return getImage(1)
        if check_name(image_name):
            try:
                global currentFile
                currentFile = image_name
                path = os.path.join("ungraded",image_name)
                r = requests.get(url, stream=True)
                if r.status_code == 200:
                    with open(path, 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)  
                image = Image.open(path)
                image.thumbnail((700,650), Image.LANCZOS)
                image.save(path)

                print(f"Progress: {x}/{amount} |",image_name," successfully downloaded.")
                
                global modelExists
                global trainingPredicted
                trainingPredicted = 0
                for x in range(len(trainingData)):
                    if os.path.exists(trainingData[x]):
                        modelExists = True
                        use_model(trainingData[x])
                        
                        trainingButtons[x].setText(f"{str(trainingData[x])}: {str(trainingPredicted)}")
                        trainingButtons[x].update()
                        trainingButtons[x].setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
                        trainingButtons[x].adjustSize()
                    else:
                        modelExists = False
                
                
            except:
                print(f"ERROR: {x}/{amount} FAILED TO DOWNLOAD!")
                print("Trying again...")
                return getImage(1)
            return path
        else:
            #dupe
            return getImage(1)

# Function to change the NSFW setting
def change_nsfw():
    global NSFW  # Access the global variable
    NSFW = nsfw_menu.currentText() == "NSFW"

# Function to change image.
def change_image(numb):
    imgPath = getImage(numb)
    for x in range(0,10):
        buttons[x].setStyleSheet("background-color: white")
    window.setStyleSheet("QMainWindow { background-color: " + bg_color+"; }")
    buttons[int(predictedFolder)-1].setStyleSheet("background-color: lime")
    pixmap = QPixmap(imgPath)
    label.setPixmap(pixmap)
    return imgPath

def sourceWeb():
    webbrowser.open(str(source))



def copyToTaceTiers():
    try:
        directory_path = QFileDialog.getExistingDirectory(taceTiersExport, 'Select tacetiers Directory')
        destination_path = os.path.join(directory_path, "wAIfuExtension")
        os.makedirs(destination_path,exist_ok=True)
        source_path = "images/"
        shutil.copytree(source_path, os.path.join(destination_path, "images"),dirs_exist_ok=True)
        os.chmod(directory_path, 0o777)
        jar_file_path = os.path.join(destination_path,"wAIfuToTaceTiers.jar")
        command = ["java", "-jar", jar_file_path]
        subprocess.call(command, cwd=destination_path)
        os.makedirs(os.path.join(os.path.join(directory_path, "Image Sets"),"wAIfu"),exist_ok=True)
        os.makedirs(os.path.join(os.path.join(os.path.join(directory_path, "Image Sets"),"wAIfu"),"img"),exist_ok=True)
        os.makedirs(os.path.join(os.path.join(directory_path,"RankingData"),"wAIfu"),exist_ok=True)
        shutil.copytree(os.path.join(destination_path,"img"), os.path.join(os.path.join(os.path.join(directory_path, "Image Sets"),"wAIfu"),"img"),dirs_exist_ok=True)
        shutil.copytree(os.path.join(os.path.join(destination_path,"RankingData"),"WaifuRater"), os.path.join(os.path.join(directory_path,"RankingData"),"wAIfu"),dirs_exist_ok=True)
        shutil.rmtree(os.path.join(destination_path,"img"))
        shutil.rmtree(os.path.join(destination_path,"RankingData"))
        shutil.rmtree(os.path.join(destination_path,"images"))
        taceTiersExport.setText("Success.")
        QDesktopServices.openUrl(QUrl.fromLocalFile(directory_path))
    except:
        taceTiersExport.setText("Failed.")

def getTaceTiers():
    webbrowser.open("https://tacecaps.itch.io/tacetier/")



# Action to clicking on buttons:
def clicked(numb):
    trainingRefresh()
    menuLayout.setEnabled(False)
    #time.sleep(0.2)
    train.setText("Train")
    global imagePath
    currentTime = time.time()
    random.seed(currentTime)
    random_int = random.randint(1,10)
    taceTiersExport.setText(taceTiersExportText)
    # write name in names.txt:
    with open(filename, "a") as file:
        file.write(currentFile + "\n")
    #Validation or training data
    if random_int <= 2:
        os.makedirs("validation", exist_ok=True)
        augmentData(os.path.join("ungraded",currentFile),os.path.join("validation",str(numb)),str(numb),random.randint(4,10))
        os.makedirs(os.path.join("validation",str(numb)), exist_ok=True) # Create folder if it doesn't exist
        image = Image.open(imagePath)
        image.thumbnail((178,218), Image.LANCZOS)
        image.save(os.path.join("validation",os.path.join(str(numb), os.path.basename(imagePath))))
        print(f"File saved in folder {str(numb)}, and saved as validation")
        if os.path.exists(imagePath):
            os.unlink(imagePath)
        imagePath = change_image(1)
    else:
        os.makedirs(os.path.join("images",str(numb)), exist_ok=True) # Create folder if it doesn't exist
        augmentData(os.path.join("ungraded",currentFile),os.path.join("images",str(numb)),str(numb),random.randint(4,10))
        image = Image.open(imagePath)
        image.thumbnail((178,218), Image.LANCZOS)
        image.save(os.path.join("images",os.path.join(str(numb), os.path.basename(imagePath))))
        print(f"File saved in folder {str(numb)}")
        if os.path.exists(imagePath):
            os.unlink(imagePath)
        imagePath = change_image(1)
    #guess.setText("Prediction: " + str(predictedFolder)+"/10")
    menuLayout.setEnabled(True)
    if modelExists == False:
        global timerFrame
        timerFrame = "point" # remember this
        timer.start(5*1000) # (1000ms = 1s)
        chibiChangeImage(5)

timerFrame = ""

def chibiTimer():
    global timerFrame
    if timerFrame == "point":
        timerFrame = "point" # remember this
        timer.start(5*1000) # (1000ms = 1s)
    chibiChangeImage(1)

def defaultClicked(numb):
    try:
        global defaultTrainingData
        file_path, _ = QFileDialog.getOpenFileName(
            parent=None,
            caption='Select an H5 file',
            filter='H5 files (*.h5)',
            initialFilter='H5 files (*.h5)'
        )
        file_info = QFileInfo(file_path)
        defaultTrainingData = file_info.fileName()
        defaultTrainingData = trainingData[numb]
        defaultButton.setText(f"Default Data: {defaultTrainingData}")
        defaultButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        defaultButton.adjustSize()
        trainingData.remove(defaultTrainingData)
        trainingData.append(defaultTrainingData)
    except:
        print("Default ejected")
    
def trainingRefresh():
    if len(trainingData) == 0:
        defaultButton.setVisible(False)
        defaultButton.setEnabled(False)
    else:
        defaultButton.setVisible(True)
        defaultButton.setEnabled(True)
    filesCheck = os.listdir()
    for file in filesCheck:
        if file.endswith(".h5") and file not in trainingData:
            # If the file is not in the array, add it
            trainingData.append(file)
            print(f"Added {file} to trainingData")
            trainingButtons.append(QPushButton(trainingData[len(trainingData)-1],window))
            trainingButtons[len(trainingData)-1].move(0,int(roomHeight/3+25*(len(trainingButtons)-1)))
            trainingButtons[len(trainingData)-1].setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            trainingButtons[len(trainingData)-1].setMinimumWidth(QFontMetrics(trainingButtons[len(trainingData)-1].font()).horizontalAdvance(trainingButtons[len(trainingData)-1].text()))
            layouta = window.layout()
            if not layouta:
                layouta = QVBoxLayout()
                window.setLayout(layout)
            layouta.addWidget(trainingButtons[len(trainingData)-1]) # Add button to layout
            window.update()


    



roomWidth = 1280
roomHeight = 720

app = QApplication(sys.argv)
app.setApplicationName("wAIfu")
window = QMainWindow()
window.setFixedSize(roomWidth, roomHeight) 
window.setGeometry(100,100,1280,720)

trainingButtons = []
for x in range(len(trainingData)):
    trainingButtons.append(QPushButton(trainingData[x],window))
    trainingButtons[x].move(0,int(roomHeight/3+25*x))
    trainingButtons[x].setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    trainingButtons[x].setMinimumWidth(QFontMetrics(trainingButtons[x].font()).horizontalAdvance(trainingButtons[x].text()))

# Creating a label to display the photo
label = QLabel(window)
imagePath = getImage(1)
window.setStyleSheet("QMainWindow { background-color: " + bg_color+"; }")
pixmap = QPixmap(imagePath)
label.setPixmap(pixmap)
label.setAlignment(Qt.AlignmentFlag.AlignCenter)

# Creating an array of buttons
buttons = []
for i in range(10):
    button = QPushButton(str(i+1), window)
    buttons.append(button)
    buttons[i].setStyleSheet("background-color: white")
buttons[int(predictedFolder)-1].setStyleSheet("background-color: lime")

# Creating a label to display the image
chibi = QLabel(window)
chibi_pixmap = QPixmap(chibi_path)
maxWidth = int(968/2.65)
maxHeight = int(1450/2.65)
chibi_pixmap = chibi_pixmap.scaled(maxWidth, maxHeight, Qt.AspectRatioMode.KeepAspectRatio)
chibi.setPixmap(chibi_pixmap)
chibi.setAlignment(Qt.AlignmentFlag.AlignRight)
chibiExists = True

# Creating a layout for the chibi image and dropdown menu
chibi_layout = QVBoxLayout()

# Adding a spacer item to position the layout at the bottom right of the window
spacer = QSpacerItem(0, -80, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # adjust the size of the spacer
chibi_layout.addItem(spacer)

# Adding the chibi image to the layout
chibi_layout.addWidget(chibi)

# Setting the alignment of the layout to bottom-right
chibi_layout.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

# Adding a spacer item to position the layout at the bottom right of the window
spacer = QSpacerItem(20, -80, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # adjust the size of the spacer
chibi_layout.addItem(spacer)

#chibitimer
timer = QTimer()
timer.timeout.connect(chibiTimer)
timerFrame = "none" # remember this


# Creating a layout for the menu items
menuLayout = QHBoxLayout()
menuLayout.addStretch(1)
for button in buttons:
    menuLayout.addWidget(button)
    menuLayout.setSpacing(10)
menuLayout.addStretch(1)


# Creating a vertical layout for the photo and buttons
photo_layout = QVBoxLayout()
photo_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
spacer = QSpacerItem(0, -320, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # adjust the size of the spacer
photo_layout.addItem(spacer)
photo_layout.addWidget(label)
spacer = QSpacerItem(0, -625-50, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # adjust the size of the spacer
photo_layout.addItem(spacer)
horizontalPhoto = QHBoxLayout()
horizontalPhoto.addLayout(photo_layout)
horizontalPhoto.setAlignment(Qt.AlignmentFlag.AlignCenter)


# Creating a horizontal layout for the photo and chibi image/dropdown layout
horizontal_layout = QHBoxLayout()
horizontal_layout.addLayout(chibi_layout)



# Creating a vertical layout for the photo, buttons, and chibi image/dropdown layout
layout = QVBoxLayout()
layout.addLayout(horizontalPhoto)
layout.addLayout(horizontal_layout)
layout.addLayout(menuLayout)



# Setting the layout to the main window
widget = QWidget(window)
widget.setLayout(layout)
window.setCentralWidget(widget)

# Bring the chibi image to the front of the other image
chibi.raise_()

# Setting the layout to the main window
window.setCentralWidget(QWidget())
window.centralWidget().setLayout(layout)

sourceButton = QPushButton("Source",window)
sourceButton.move(0,roomHeight-25)

taceTiersExportText = "Export: tacetier"
taceTiersExport = QPushButton(taceTiersExportText,window)
taceTiersExport.move(roomWidth-100,0)

taceTiersGet = QPushButton("Get tacetier",window)
taceTiersGet.move(roomWidth-100,25)

sourceButton.clicked.connect(sourceWeb)


taceTiersExport.clicked.connect(copyToTaceTiers)

taceTiersGet.clicked.connect(getTaceTiers)

# Creating Default Button:
defaultButton = QPushButton(f"Default Data: {defaultTrainingData}",window)
defaultButton.move(0,int(roomHeight/3-25))
defaultButton.clicked.connect(defaultClicked)
defaultButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
defaultButton.adjustSize()
defaultButton.setStyleSheet("background-color: lime")
#trainingRefresh()


# Creating a dropdown menu for NSFW/SFW
nsfw_menu = QComboBox(window)
nsfw_menu.addItem("SFW")
if version == "NSFW":
    nsfw_menu.addItem("NSFW")
else:
    nsfw_menu.addItem("No NSFW Access")
nsfw_menu.setFixedSize(100, 30)
nsfw_menu.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

# Connect the dropdown menu to the change_nsfw function
nsfw_menu.currentIndexChanged.connect(change_nsfw)

# Button Actions
for i in range(len(buttons)):
    buttons[i].clicked.connect(lambda state, x=i: clicked(x+1))


# Creating Train Button:
train = QPushButton("Train",window)
train.move(25*4,0)
train.clicked.connect(trainClicked)

window.show()

sys.exit(app.exec())