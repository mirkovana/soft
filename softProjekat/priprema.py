import cv2
import numpy as np
import matplotlib.pyplot as plt
# keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.models import model_from_json
import os
import display_size as ds
from scipy import signal
import glob
import sys

train_image_paths = '.'+os.path.sep+'dataset'+os.path.sep+'trening'+os.path.sep+'Digital-Numbers - Copy.png'
train_image_paths2 = '.'+os.path.sep+'dataset'+os.path.sep+'trening'+os.path.sep+'Digital-Numbers - Copy (2).png'
train_image_paths3 = '.'+os.path.sep+'dataset'+os.path.sep+'trening'+os.path.sep+'digital-number-3.png'

#test_image_path = '.'+os.path.sep+'dataset'+os.path.sep+'validation'+os.path.sep+'1f4450408639279059c3ad808a0e37f03d0fa14b.jpg'
TRAINED_FOLDER_PATH = '.'+os.path.sep+'trained_model'+os.path.sep
def select_roi2(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_orig, contours, -1, (255, 0, 0), 1)
    # print("BROJ KONTURA 2 ", len(contours))
    #display_image(image_orig)
    sorted_regions = []  # lista sortiranih regiona po X osi
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if h>1.9*visina_displeja/3 and h<=visina_displeja and w<290:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda x: x[1][0])

    sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions


def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#za trening
def image_bin1(image_gs):
    imgray = cv2.equalizeHist(image_gs)
    ret,image_bin = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY)

    return image_bin

#za testiranje
def image_bin2(image_gs):
    imgray = cv2.equalizeHist(image_gs)
    ret,image_bin = cv2.threshold(imgray, 25, 255, cv2.THRESH_BINARY)
    #image_bin = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)
    #otsu_threshold, image_bin = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #display_image(image_bin)

    return image_bin

#za trening
def dilate(image):
    kernel = np.ones((7,7))
    return cv2.dilate(image, kernel, iterations=1)

#za testiranje
def dilate2(image):
    kernel = np.ones((7,7))
    return cv2.dilate(image, kernel, iterations=1)

#za trening
def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=5)

#za testiranje
def erode2(image):
    kernel = np.ones((2, 2))
    return cv2.erode(image, kernel, iterations=11)

#za trening
def erode3(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=10)

def resize_region(region):
    resized = cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)
    return resized

def scale_to_range(image):
    return image / 255

def matrix_to_vector(image):
    return image.flatten()

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann

def convert_output(outputs):
    return np.eye(len(outputs))

def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(30, activation='sigmoid'))
    return ann

def train_ann(ann, x_train, y_train):
    x_train = np.array(x_train, np.float32)
    y_train = np.array(y_train, np.float32)
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    ann.fit(x_train, y_train, epochs=3000, batch_size=1, verbose=0, shuffle=False)

    return ann

def select_roi(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_orig, contours, -1, (255, 0, 0), 1)
    # print("BROJ KONTURA", len(contours))
    #display_image(image_orig)
    sorted_regions = []  # lista sortiranih regiona po X osi
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        #if area > 100 and h < 100 and h > 65 and w > 11:
        if area > 100 and h < 120 and h > 65 and w > 11:
            if w<30:
                w=46
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda x: x[1][0])

    sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions


def load_or_train_ann(train_image_paths,train_image_paths2,train_image_paths3, trained_folder):
    image_color = load_image(train_image_paths)
    # display_image(image_color)
    img = image_bin1(image_gray(image_color))
    dil1 = dilate(img)

    image_erode = erode(dil1)
    # display_image(image_erode)

    selected_regions, letters = select_roi(image_color.copy(), image_erode)
    # trening dva ---------------------------------------------------------------------
    image_color1 = load_image(train_image_paths2)
    # display_image(image_color)
    img1 = image_bin1(image_gray(image_color1))
    dil1 = dilate(img1)

    image_erode1 = erode3(dil1)
    # display_image(image_erode)

    selected_regions1, letters1 = select_roi(image_color1.copy(), image_erode1)
    # trening tri---------------------------------------------------------------------
    image_color3 = load_image(train_image_paths3)
    # display_image(image_color)
    img3 = image_bin1(image_gray(image_color3))
    dil3 = dilate(img3)

    image_erode3 = erode(dil3)

    selected_regions3, letters3 = select_roi(image_color3.copy(), image_erode3)

    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # for let in letters:
    #    display_image(let)
    # for let in letters1:
    #     display_image(let)
    # for let in letters3:
    #     display_image(let)
    inputs = prepare_for_ann(letters+letters1+letters3)

    outputs = convert_output(alphabet)

    try:
        json_file = open(trained_folder + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        ann.load_weights(trained_folder + "model.h5")
        print("Istrenirani model uspesno ucitan.")
    except Exception as e:
        ann = None

    if ann == None:
        print("Traniranje modela zapoceto.")
        ann = create_ann()
        ann = train_ann(ann, inputs, outputs)
        print("Treniranje modela zavrseno.")
        model_json = ann.to_json()
        with open(trained_folder + "model.json", "w") as json_file:
            json_file.write(model_json)
        ann.save_weights(trained_folder + "model.h5")

    return ann


def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def extract_text_from_image(trained_model, image_path):
    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    image_color = load_image(image_path)

    img_gray = image_gray(image_color)
    # display_image(img_gray)
    img = image_bin2(image_gray(image_color))

    image_erode = erode2(img)
    # display_image(image_erode)

    dil1 = dilate2(image_erode)

    selected_regions, letters = select_roi2(image_color.copy(), dil1)
    # display_image(selected_regions)

    # for let in letters:
    #    display_image(let)

    rez=[]
    res = ''
    if len(letters)>0 : #da ne bi pucao porgram kada ne nadje displej
        test_inputs = prepare_for_ann(letters)
        result = ann.predict(np.array(test_inputs, np.float32))
        print(display_result(result, alphabet))
        rez = display_result(result, alphabet)
        for r in rez:
            res+=r
    return rez

if __name__ == '__main__':
    print("POCETAK")
    ann = load_or_train_ann(train_image_paths, train_image_paths2,train_image_paths3, TRAINED_FOLDER_PATH)
    visina_displeja = 0
    sirina_displeja = 0
    # visina_displeja, sirina_displeja = ds.display_size(test_image_path)
    # print("sirina i visina" , visina_displeja)
    # extract_text_from_image(ann, test_image_path)

    #---------------------------------------------EVALUACIJA--------------------------------------------------
    if len(sys.argv) > 1:
        VALIDATION_DATASET_PATH = sys.argv[1]
    else:
        VALIDATION_DATASET_PATH = '.' + os.path.sep + 'dataset' + os.path.sep+'validation'+os.path.sep

    processed_image_names = []
    extracted_text = []

    for image_path in glob.glob(VALIDATION_DATASET_PATH + "*.jpg"):
        visina_displeja, sirina_displeja = ds.display_size(image_path)
        image_directory, image_name = os.path.split(image_path)
        processed_image_names.append(image_name)
        # extracted_text.append(extract_text_from_image(ann, image_path))
        result = extract_text_from_image(ann, image_path)
        r = ''
        for res in result:
             r += res
        extracted_text.append(r)

    result_file_contents = ""
    for image_index, image_name in enumerate(processed_image_names):
        result_file_contents += "%s,%s\n" % (image_name, extracted_text[image_index])

    with open('result.csv', 'w', encoding='utf-8') as output_file:
        output_file.write(result_file_contents)
    print("KRAJ")