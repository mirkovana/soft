import cv2
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
# keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
import os
from scipy import signal
import display_size as ds
import matplotlib.image as mpimg
from scipy import signal
#C:\Users\Marija\Desktop\soft\softProjekat\dataset\0a32664b2b8c8fe725994dfaab1d9a998dd0df09.jpg
# Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans
#train_image_paths = '.'+os.path.sep+'dataset'+os.path.sep+'trening'+os.path.sep+'Digital-Numbers.png'
train_image_paths = r'C:\Users\Marija\Desktop\soft\softProjekat\dataset\trening\Digital-Numbers - Copy.png'
train_image_paths2 = r'C:\Users\Marija\Desktop\soft\softProjekat\dataset\trening\Digital-Numbers - Copy (2).png'
train_image_paths3 = r'C:\Users\Marija\Desktop\soft\softProjekat\dataset\trening\digital-number-3.png'
#train_image_paths = r'C:\Users\Ana\Desktop\soft\softProjekat\dataset\trening\Digital-Numbers - Copy.png'
#test_image_path = r'C:\Users\Ana\Desktop\soft\softProjekat\dataset\0a1e1c676aa31a9f56818e580d7a2e20689fefba.jpg'
test_image_path = '.'+os.path.sep+'dataset'+os.path.sep+'73bdc1e381510f46aac391bddd99d2dee1f39e8d.jpg'

SERIALIZATION_FOLDER_PATH = '.'+os.path.sep+'serialized_model'+os.path.sep
def select_roi2(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''

    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_orig, contours, -1, (255, 0, 0), 1)
    # print("BROJ KONTURA 2 ", len(contours))
    display_image(image_orig)
    sorted_regions = []  # lista sortiranih regiona po X osi
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        # print("NAS X", x)
        # print("NAS Y", y)
        # print("NAS W", w)
        print("NAS H", h)
        area = cv2.contourArea(contour)
        #print("AREAAAA", area)
        if h>visina_displeja/2 and h<visina_displeja:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    #za izbacivanje
    #outer_regions = remove_inner_region(regions_array)
    #crtkaj = draw_rectangles(image_orig, outer_regions)
    #display_image(crtkaj)
    sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions
def display_image(image, color=False):
    # imgplot = plt.imshow(image)
    # plt.show()
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
def image_blur(img_gray):
    k_size = 5
    k = (1. / k_size * k_size) * np.ones((k_size, k_size))
    image_blur = signal.convolve2d(img_gray, k)
    plt.imshow(image_blur, 'gray')

    return image_blur

def image_bin1(image_gs):
    imgray = cv2.equalizeHist(image_gs)  # global
    # plt.imshow(imgray, 'gray')
    # plt.show()
    ret,image_bin = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY)

    return image_bin

def image_bin2(image_gs):
    imgray = cv2.equalizeHist(image_gs)  # global
    ret,image_bin = cv2.threshold(imgray, 25, 255, cv2.THRESH_BINARY)
    #image_bin = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)
    #otsu_threshold, image_bin = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    display_image(image_bin)
    #print("Obtained threshold: ", otsu_threshold)

    return image_bin

def dilate(image):
    kernel = np.ones((7,7))
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=5)

def erode2(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=5)

def invert(image):
    return 255 - image

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
    # print("X je ove duzinee : %d" % len(X_train))
    # print("Y je ove duzinee : %d" % len(y_train))
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    ann.fit(x_train, y_train, epochs=3000, batch_size=1, verbose=0, shuffle=False)

    return ann

def draw_rectangles(image_orig, regions):
    for region in regions:
        x, y, w, h = region[1]
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2) #iscrtavanje nasih pravougaonika na originalnoj slici
    return image_orig

def remove_inner_region(sorted_regions):
    idx = 0
    outer_regions = []
    while idx < (len(sorted_regions) - 1):
        current_rect = sorted_regions[idx][1]
        next_rect = sorted_regions[idx + 1][1]
        if (current_rect[0] - next_rect[0] < 6):
            new_rect = (current_rect[0], next_rect[1],
                        current_rect[2], current_rect[3] + (current_rect[1] - next_rect[1]))
            new_figure = sorted_regions[idx][0] + sorted_regions[idx + 1][0]
            outer_regions.append([new_figure, new_rect])
            idx += 1
    return outer_regions

def select_roi(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''

    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_orig, contours, -1, (255, 0, 0), 1)
    # print("BROJ KONTURA", len(contours))
    #display_image(image_orig)
    sorted_regions = []  # lista sortiranih regiona po X osi
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        # print("NAS X", x)
        # print("NAS Y", y)
        # print("NAS W", w)
        # print("NAS H", h)
        area = cv2.contourArea(contour)
        #print("AREAAAA", area)
        #if area > 100 and h < 100 and h > 65 and w > 11:
        if area > 100 and h < 120 and h > 65 and w > 11:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
            if w<30:
                w=46
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    #za izbacivanje
    #outer_regions = remove_inner_region(regions_array)
    #crtkaj = draw_rectangles(image_orig, outer_regions)
    #display_image(crtkaj)
    sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions




def train_or_load_character_recognition_model(train_image_paths,train_image_paths2,train_image_paths3, serialization_folder):
    """
       Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta), kao i
       putanju do foldera u koji treba sacuvati model nakon sto se istrenira (da ne trenirate svaki put iznova)

       Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

       Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
       istreniran i ako se nalazi u folderu za serijalizaciju

       :param train_image_paths: putanje do fotografija alfabeta
       :param serialization_folder: folder u koji treba sacuvati serijalizovani model
       :return: Objekat modela
       """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    image_color = load_image(train_image_paths)
    # display_image(image_color)
    img = image_bin1(image_gray(image_color))
    dil1 = dilate(img)

    image_erode = erode(dil1)
    # display_image(image_erode)
    # plt.imshow(image_erode, 'gray')
    # plt.show()

    inv = invert(image_erode)
    # plt.imshow(inv, 'gray')
    # plt.show()

    selected_regions, letters = select_roi(image_color.copy(), image_erode)
    #dvojka---------------------------------------------------------------------
    image_color1 = load_image(train_image_paths2)
    # display_image(image_color)
    img1 = image_bin1(image_gray(image_color1))
    dil1 = dilate(img1)

    image_erode1 = erode(dil1)
    # display_image(image_erode)
    # plt.imshow(image_erode, 'gray')
    # plt.show()

    inv1 = invert(image_erode1)
    # plt.imshow(inv, 'gray')
    # plt.show()

    selected_regions1, letters1 = select_roi(image_color1.copy(), image_erode1)
    # trojka---------------------------------------------------------------------
    image_color3 = load_image(train_image_paths3)
    # display_image(image_color)
    img3 = image_bin1(image_gray(image_color3))
    dil3 = dilate(img3)

    image_erode3 = erode(dil3)
    # display_image(image_erode)
    # plt.imshow(image_erode, 'gray')
    # plt.show()

    inv1 = invert(image_erode1)
    # plt.imshow(inv, 'gray')
    # plt.show()

    selected_regions3, letters3 = select_roi(image_color3.copy(), image_erode3)

    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #print("SR", selected_regions)
    #display_image(selected_regions)
    #print("leee", letters)
    # for let in letters:
    #    display_image(let)
    # for let in letters1:
    #     display_image(let)
    # for let in letters3:
    #     display_image(let)
    inputs = prepare_for_ann(letters+letters1+letters3)

    outputs = convert_output(alphabet)

    try:
        json_file = open(serialization_folder + 'neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        ann.load_weights(serialization_folder + "neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
    except Exception as e:
        ann = None

    if ann == None:
        print("Traniranje modela zapoceto.")
        ann = create_ann()
        ann = train_ann(ann, inputs, outputs)
        print("Treniranje modela zavrseno.")
        model_json = ann.to_json()
        with open(serialization_folder + "neuronska.json", "w") as json_file:
            json_file.write(model_json)
        ann.save_weights(serialization_folder + "neuronska.h5")



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
    # display_image(img)
    #
    # slika_blur = image_blur(img)
    # display_image(slika_blur)
    #dil1 = dilate(img)
    #display_image(dil1)

    image_erode = erode2(img)
    # display_image(image_erode)

    dil1 = dilate(image_erode)
    display_image(dil1)
    # rgb_planes = cv2.split(image_color)
    #
    # result_planes = []
    # result_norm_planes = []
    # for plane in rgb_planes:
    #     dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
    #     bg_img = cv2.medianBlur(dilated_img, 5)
    #     diff_img = 255 - cv2.absdiff(plane, bg_img)
    #     norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    #     result_planes.append(diff_img)
    #     result_norm_planes.append(norm_img)
    #
    # result = cv2.merge(result_planes)
    # result_norm = cv2.merge(result_norm_planes)
    # display_image(result)
    # display_image(result_norm)
    inv = invert(dil1)
    # display_image(inv)

    selected_regions, letters = select_roi2(image_color.copy(), dil1)
    display_image(selected_regions)

    # for let in letters:
    #   display_image(let)

    test_inputs = prepare_for_ann(letters)
    result = ann.predict(np.array(test_inputs, np.float32))
    print(display_result(result, alphabet))

if __name__ == '__main__':
    print("SAD")
    ann = train_or_load_character_recognition_model(train_image_paths, train_image_paths2,train_image_paths3, SERIALIZATION_FOLDER_PATH)
    visina_displeja, sirina_displeja = ds.display_size(test_image_path)
    print("sirina i visina" , visina_displeja)
    extract_text_from_image(ann, test_image_path)
    print("KRAJ")