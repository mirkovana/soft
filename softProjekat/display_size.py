import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def resize_region(region):
    resized = cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)
    return resized

def invert(image):
    return 255 - image

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=5)

def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)

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
    #otsu_threshold, image_bin = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, image_bin = cv2.threshold(imgray, 70, 255, cv2.THRESH_BINARY)
    #display_image(image_bin)
    return image_bin


def display_image(image, color=False):
    # imgplot = plt.imshow(image)
    # plt.show()
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()

def display_size(train_image_paths):
    image_color = load_image(train_image_paths)
    #display_image(image_gray(image_color))
    img = image_bin1(image_gray(image_color))
    #display_image(img)
    dil1 = dilate(img)
    #display_image(dil1)
    image_erode = erode(dil1)
    # display_image(image_erode)
    # plt.imshow(image_erode, 'gray')
    # plt.show()

    inv = invert(image_erode)
    # plt.imshow(inv, 'gray')
    # plt.show()

    povratna_visina, sirina_povratna = select_roi(image_color.copy(), image_erode)
    return povratna_visina, sirina_povratna


def select_roi(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    v = np.median(image_bin)
    sigma = 0.35
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edge =cv2.Canny(image_bin, lower, upper)

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_orig, contours, -1, (255, 0, 0), 1)
    print("BROJ KONTURA", len(contours))
    #display_image(image_orig)
    regions_array = []
    sirine = []
    visine = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        sirine.append(w)
        visine.append(h)
        region = image_bin[y:y + h + 1, x:x + w + 1]
        regions_array.append([resize_region(region), (x, y, w, h)])
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    sorted_regions = [region[0] for region in regions_array]
    #display_image(image_orig)
    visine.sort(reverse=True)
    sirine.sort(reverse=True)
    povratna_visina = visine[0]
    sirina_povratna = sirine[0]
    return povratna_visina, sirina_povratna