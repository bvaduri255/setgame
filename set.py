import cv2 as cv
import os
import numpy as np
import imutils
import random
from matplotlib import pyplot as plt

img = cv.imread('imgs/img3.png')

window_name = 'Image'
image_shape = img.shape
card_num = 0
if image_shape[0] < 700:
    card_num = 12
    splitter_dict = {'1' : [[0,0],[255,156]], '2' : [[255,0],[501,156]], '3' : [[501,0],[image_shape[1],156]], '4' : [[0,156],[255,310]], '5' : [[255,156],[501,310]], '6' : [[501,156],[image_shape[1],310]], '7' : [[0,310],[255,466]], '8' : [[255,310],[501,466]], '9' : [[501,310],[image_shape[1],466]], '10' : [[0,466],[255,image_shape[0]]], '11' : [[255,466],[501,image_shape[0]]], '12' : [[501,466],[image_shape[1],image_shape[0]]]} 
elif image_shape[0] < 850:
    card_num = 15
    splitter_dict = {'1' : [[0,0],[255,156]], '2' : [[255,0],[501,156]], '3' : [[501,0],[image_shape[1],156]], '4' : [[0,156],[255,310]], '5' : [[255,156],[501,310]], '6' : [[501,156],[image_shape[1],310]], '7' : [[0,310],[255,466]], '8' : [[255,310],[501,466]], '9' : [[501,310],[image_shape[1],466]], '10' : [[0,466],[255,622]], '11' : [[255,466],[501,622]], '12' : [[501,466],[image_shape[1],622]], '13' : [[0,622],[255,image_shape[0]]], '14' : [[255,622],[501,image_shape[0]]], '15' : [[501,622],[image_shape[1],image_shape[0]]]}
elif image_shape[0] < 1000:
    card_num = 18
    # Code Later
else:
    card_num = 21
    # Code Later

print(card_num)


def splitter(image):
    name = 'Slice'
    file_path = r"tempimgs/"
    for i in range(1,card_num + 1):
        i_str = str(i)
        temp_name = name  + i_str
        x2 = splitter_dict[i_str][1][0]
        y2 = splitter_dict[i_str][1][1]
        x1 = splitter_dict[i_str][0][0]
        y1 = splitter_dict[i_str][0][1]
        temp_img = image[y1 : y2, x1 : x2].copy()
        #temp_img = image[x1 : x2, y1 : y2].copy()
        cv.imwrite(r"tempimgs/" + temp_name + ".png" , temp_img)

    print('Done!')

def ternary(n):
    e = n // 3
    q = n % 3
    if n == 0:
        return 0
    elif e == 0:
        return str(q)
    else:
        return ternary(e) + str(q)

def int_to_list(n):
    n = str(n)
    while len(n) < 4:
        n = '0' + n
    
    out = [int(x) for x in str(n)]

    return out

def make_card_dict():
    listy = [0,0,0,0]
    card_dict = {'1' : [0,0,0,0]}

    for i in range(2,82):
        num = ternary(i-1)
        listy = int_to_list(num)
        temp_listy = listy.copy()
        card_dict[str(i)] = temp_listy
    
    return card_dict

def color(img):
    img1 = img.copy()
    
def histogram_compare(img1, img2):
    hsv_img1 = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
    hsv_img2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)

    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges 
    
    channels = [0, 1]

    hist_img1 = cv.calcHist([hsv_img1], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    hist_img2 = cv.calcHist([hsv_img2], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    img1_img2 = cv.compareHist(hist_img1, hist_img2, method=cv.HISTCMP_CHISQR_ALT)

    return img1_img2

def template_match(img, template):
    img1 = img.copy()
    method = ['cv.TM_CCOEFF_NORMED']
    max_value = 0
    for meth in method:
        img = img1.copy()
        method = eval(meth)

        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        max_value = max(max_value, max_val)

    return max_value

def template_match2(img, template):
    template = cv.Canny(template, 10, 25)
    (height, width) = template.shape[:2]
    temp_found = None
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized_img = imutils.resize(img, width = int(img.shape[1] * scale))
        ratio = img.shape[1] / float(resized_img.shape[1])
        if resized_img.shape[0] < height or resized_img.shape[1] < width:
            break
        e = cv.Canny(resized_img, 10, 25)
        match = cv.matchTemplate(e, template, cv.TM_CCOEFF_NORMED)
        (_, val_max, _, loc_max) = cv.minMaxLoc(match)
        if temp_found is None or val_max>temp_found[0]:
            temp_found = (val_max, loc_max, ratio)

    (val_max, loc_max, r) = temp_found

    return val_max

def number_buffer(x):
    x = str(x)
    if len(x) == 1:
        x = '0' + x

    return x

def card_to_vect():
    card_vect_list = []
    for i in range(1, card_num + 1):
        file_name = 'tempimgs/' + 'Slice'+str(i)+'.png'
        print(file_name)
        curr_img = cv.imread(file_name)
        curr_img = cv.resize(curr_img, (300,175))
        max_val, max_ind = 0, 0
        for j in range(1,82):
            curr_card_name = 'allcards/' + 'card' + number_buffer(j) + '.PNG'
            print(curr_card_name)
            curr_card = cv.imread(curr_card_name)
            curr_card = cv.resize(curr_card, (300,175))
            curr_card_trim = curr_card[25:150, 35 : 265]
            curr_hc = template_match2(curr_img, curr_card_trim)
            if max(max_val, curr_hc) == curr_hc:
                max_val = max(max_val, curr_hc)
                max_ind = j
        #card_vect_list.append(int_to_list(ternary(max_ind)))
        card_vect_list.append(max_ind)
    
    return card_vect_list

'''
splitter(img)
make_card_dict()
print(card_to_vect())'''