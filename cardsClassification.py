import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import shutil
import os
from flask import Flask
from dotenv import load_dotenv
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image

load_dotenv('var.env')
servertestdir = os.environ['SERVERTESTDIR']
modeldir = os.environ['MODELDIR']
labelsdir = os.environ['LABELSDIR']

app = Flask(__name__)

def flip_image(image,dir,folder_name,extension):
    image = cv2.flip(image, dir)
    cv2.imwrite(folder_name + "/flip-" + str(dir)+extension, image)


def add_light(image,gamma,folder_name,extension):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(folder_name + "/light-"+str(gamma)+extension, image)
    else:
        cv2.imwrite(folder_name + "/dark-" + str(gamma) + extension, image)


def saturation_image(image,saturation,folder_name,extension):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(folder_name + "/saturation-" + str(saturation) + extension, image)


def contrast_image(image,contrast,folder_name,extension):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
    image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(folder_name + "/Contrast-" + str(contrast) + extension, image)


def rotate_image(image,deg,folder_name,extension):
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(folder_name + "/Rotate-" + str(deg) + extension, image)


def translation_image(image,x,y,folder_name,extension):
    rows, cols ,c= image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(folder_name + "/Translation-" + str(x) + str(y) + extension, image)


def gausian_blur(image,blur,folder_name,extension):
    image = cv2.GaussianBlur(image,(5,5),blur)
    cv2.imwrite(folder_name+"/GausianBLur-"+str(blur)+extension, image)


def averageing_blur(image,shift,folder_name,extension):
    image=cv2.blur(image,(shift,shift))
    cv2.imwrite(folder_name + "/AverageingBLur-" + str(shift) + extension, image)


def median_blur(image,shift,folder_name,extension):
    image=cv2.medianBlur(image,shift)
    cv2.imwrite(folder_name + "/MedianBLur-" + str(shift) + extension, image)


def bileteralBlur(image,d,color,space,folder_name,extension):
    image = cv2.bilateralFilter(image, d,color,space)
    cv2.imwrite(folder_name + "/BileteralBlur-"+str(d)+"*"+str(color)+"*"+str(space)+ extension, image)


def erosion_image(image, shift, folder_name, extension=".jpg"):
    kernel = np.ones((shift, shift), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    file_name = f"Erosion-{shift}{extension}"
    file_path = os.path.join(folder_name, file_name)
    cv2.imwrite(file_path, eroded_image)


def dilation_image(image, shift, folder_name, extension=".jpg"):
    kernel = np.ones((shift, shift), np.uint8)
    dilate_image = cv2.dilate(image, kernel, iterations=1)
    file_name = f"Dilate-{shift}{extension}"
    file_path = os.path.join(folder_name, file_name)
    cv2.imwrite(file_path, dilate_image)


def opening_image(image, shift, folder_name, extension=".jpg"):
    kernel = np.ones((shift, shift), np.uint8)
    opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    file_name = f"Opening-{shift}{extension}"
    file_path = os.path.join(folder_name, file_name)
    cv2.imwrite(file_path, opening_image)


def transformation_image(image,folder_name,extension):
    rows, cols, ch = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(folder_name + "/Transformations-" + str(1) + extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [0, 150]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(folder_name + "/Transformations-" + str(2) + extension, image)


def morphological_gradient_image(image, shift, folder_name, extension=".jpg"):
    kernel = np.ones((shift, shift), np.uint8)
    morphological_gradient_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    file_name = f"morphological-{shift}{extension}"
    file_path = os.path.join(folder_name, file_name)
    cv2.imwrite(file_path, morphological_gradient_image)


def top_hat_image(image, shift, folder_name, extension=".jpg"):
    kernel = np.ones((shift, shift), np.uint8)
    top_hat_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    file_name = f"THI-{shift}{extension}"
    file_path = os.path.join(folder_name, file_name)
    cv2.imwrite(file_path, top_hat_image)


def add_gaussian_noise(image,shift,mean,std_dev,folder_name,extension):
    # Generate random Gaussian noise
    rows, cols, channels = image.shape
    noise = np.random.normal(mean, std_dev, (rows, cols, channels))
    # Add the noise to the image
    noisy_image = image + noise
    # Clip the pixel values to the range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    file_name = f"AddGausNoice-{shift}{extension}"
    file_path = os.path.join(folder_name, file_name)
    cv2.imwrite(file_path, noisy_image)
    

# def randomRemove():
    


def cutout(image,shift,x, y, w, h,folder_name,extension):
    x, y, w, h = x, y, w, h
    roi = image[y:y+h, x:x+w]
    file_name = f"Cutout-{shift}{extension}"
    file_path = os.path.join(folder_name, file_name)
    resized_image = cv2.resize(roi, (640, 640))
    cv2.imwrite(file_path, resized_image)
    

def sharpen_image(image,folder_name,extension):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(folder_name+"/Sharpen-"+extension, image)


def addeptive_gaussian_noise(image,folder_name,extension):
    h,s,v=cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image=cv2.merge([h,s,v])
    cv2.imwrite(folder_name + "/Addeptive_gaussian_noise-" + extension, image)


def grayscale_image(image,folder_name,extension):
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(folder_name + "/Grayscale-" + extension, image)


def data_augmentation(input_file):
    #read image
    image=cv2.imread('/home/ubuntu/Flask_API/Cards/' + input_file)
    output_folder, extension = os.path.splitext(input_file)

    #creating folder
    if not os.path.exists("/home/ubuntu/Flask_API/Dataset" + "/" + output_folder):
        os.makedirs("/home/ubuntu/Flask_API/Dataset" + "/" + output_folder)

    #resizing image
    output_file = output_folder + extension
    size = (640, 640)

    # Open input image file
    with Image.open('/home/ubuntu/Flask_API/Cards/' +input_file) as im:
        # Resize image
        im_resized = im.resize(size)
        # Save resized image
        im_resized.save("/home/ubuntu/Flask_API/Dataset" + "/" + output_folder + "/" + output_file)

    image_file="/home/ubuntu/Flask_API/Dataset" + "/" + output_folder + "/" + output_file
    image=cv2.imread(image_file)
    
    folder_name = "/home/ubuntu/Flask_API/Dataset" + "/" + output_folder
    #Function Calling
    flip_image(image,0,folder_name,extension)#horizontal
    flip_image(image,1,folder_name,extension)#vertical
    flip_image(image,-1,folder_name,extension)#both

    add_light(image,1.5,folder_name,extension)
    add_light(image,2.0,folder_name,extension)
    add_light(image,2.5,folder_name,extension)
    add_light(image,0.7,folder_name,extension)
    add_light(image,0.4,folder_name,extension)
    add_light(image,0.3,folder_name,extension)
    add_light(image,0.1,folder_name,extension)

    saturation_image(image,50,folder_name,extension)
    saturation_image(image,100,folder_name,extension)

    contrast_image(image,25,folder_name,extension)
    contrast_image(image,50,folder_name,extension)
    contrast_image(image,100,folder_name,extension)

    rotate_image(image,90,folder_name,extension)
    rotate_image(image,-30,folder_name,extension)
    rotate_image(image,-90,folder_name,extension)
    rotate_image(image,120,folder_name,extension)
    rotate_image(image,150,folder_name,extension)
    rotate_image(image,180,folder_name,extension)
    rotate_image(image,220,folder_name,extension)
    rotate_image(image,270,folder_name,extension)

    translation_image(image,150,150,folder_name,extension)
    translation_image(image,-150,150,folder_name,extension)
    translation_image(image,150,-150,folder_name,extension)
    translation_image(image,-150,-150,folder_name,extension)

    gausian_blur(image,0.25,folder_name,extension)
    gausian_blur(image,0.50,folder_name,extension)
    gausian_blur(image,1,folder_name,extension)
    gausian_blur(image,2,folder_name,extension)
    gausian_blur(image,4,folder_name,extension)

    flip_image(image,1,folder_name,extension)#vertical

    averageing_blur(image,5,folder_name,extension)
    averageing_blur(image,4,folder_name,extension)
    averageing_blur(image,6,folder_name,extension)

    median_blur(image,3,folder_name,extension)
    median_blur(image,5,folder_name,extension)
    median_blur(image,7,folder_name,extension)

    bileteralBlur(image,9,75,75,folder_name,extension)
    bileteralBlur(image,12,100,100,folder_name,extension)
    bileteralBlur(image,25,100,100,folder_name,extension)
    bileteralBlur(image,40,75,75,folder_name,extension)

    erosion_image(image, 1, folder_name, extension)
    erosion_image(image, 3, folder_name, extension)
    erosion_image(image, 7, folder_name, extension)

    dilation_image(image, 1, folder_name, extension)
    dilation_image(image, 3, folder_name, extension)
    dilation_image(image, 5, folder_name, extension)

    opening_image(image, 1, folder_name, extension)
    opening_image(image, 3, folder_name, extension)
    opening_image(image, 4, folder_name, extension)

    transformation_image(image,folder_name,extension)
    morphological_gradient_image(image, 8, folder_name, extension)
    
    top_hat_image(image, 200, folder_name, extension)
    top_hat_image(image, 300, folder_name, extension)
    top_hat_image(image, 500, folder_name, extension)
    
    add_gaussian_noise(image,3,0,50,folder_name,extension)
    add_gaussian_noise(image,4,0,100,folder_name,extension)
    
    cutout(image,1,100,200,300,400,folder_name,extension)
    cutout(image,2,200,300,400,500,folder_name,extension)
    cutout(image,3,400,300,200,100,folder_name,extension)
    
    sharpen_image(image,folder_name,extension)
    addeptive_gaussian_noise(image,folder_name,extension)
    grayscale_image(image,folder_name,extension)


def train_model():
    # Define data paths
    train_dir = '/home/ubuntu/Flask_API/Dataset'

    # Define image size and batch size
    img_size = (224, 224)
    batch_size = 16

    # Create ImageDataGenerator objects for data augmentation and normalization
    train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    shear_range=0.2,
                                    fill_mode='nearest')

    # Create generator objects for training
    train_data = train_datagen.flow_from_directory(train_dir,
                                                target_size=img_size,
                                                batch_size=batch_size,
                                                class_mode='categorical')
    
    train_labels = train_data.classes
    # Load the pre-trained InceptionV3 model without the top layer
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom top layer for our specific task
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(len(set(train_data.classes)), activation='softmax'))

    # Create the optimizer with the new learning rate
    opt = Adam(learning_rate=0.001)

    # Compile the model with appropriate loss and optimizer for our task
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Train the model on our dataset
    model.fit(train_data, epochs=50, batch_size=16)

    # Save the model to a file
    model.save(MODELDIR)

    # Get the class indices and labels
    class_indices = train_data.class_indices
    labels = list(class_indices.keys())

    # Save the labels to a text file
    with open(LABELSDIR, 'w') as f:
        for label in labels:
            f.write(label + '\n')


def comapre_images(testImg): 
    np.set_printoptions(suppress=True)
    model = load_model(str(modeldir), compile=False)
    class_names = open(str(labelsdir), 'r').readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(str(servertestdir) + '/' + testImg).convert('RGB') 
    image_array = np.asarray(ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS))
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = (prediction[0][index])*100

    os.remove(str(servertestdir) + '/' + testImg)

    if confidence_score >= 60:
        class_name = class_name.replace("\n","")
        return class_name
    else:
        return "Please Try Again..."


@app.route('/CompareImage/<imgName>')
def result(imgName):
    return comapre_images(imgName)


@app.route('/TrainModel/<imgName>')
def modelTraining(imgName):
    data_augmentation(imgName)
    train_model()
    return "Model Training..."


if __name__ == "__main__":
    # data_augmentation("0 AUSSP.png")
    app.run()
