import cv2 as cv
import os
import numpy as np
import imutils
import random
import math
from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
from matplotlib import pyplot as plt

img = cv.imread('imgs/img3.png')
white = [255,255,255]
black = [0,0,0]
green = [25,127,21]
red = [0,0,253]
purple = [126,11,126]

color_dict = {'WHITE' : [255,255,255], 'BLACK' : [0,0,0], 'GREEN' : [21,127,25], 'RED' : [253,7,7], 'PURPLE' : [129,18,129]}

red_hex = ['#fd0707', '#fd0808','#fd0101','#fd0303']
purple_hex = ['#811281','#7e0b7e']
green_hex = ['#157f19','#518653','#167f1a','#19811d']

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

def color(img,x,y):
    out = img[y,x]

    return out

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    
def dist(arr1,arr2):
    dx = arr1[0]-arr2[0]
    dy = arr1[1]-arr2[1]
    dz = arr1[2]-arr2[2]
    out = math.sqrt((dx**2)+(dy**2)+(dz**2))

    return out

def get_image(image_path):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image

def get_colors(image, number_of_colors):
    
    modified_image = cv.resize(image, (600, 400), interpolation = cv.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    return hex_colors

def color_dist(hex):
    COLORS = ['GREEN','RED','PURPLE']
    dist_list = []
    ind = 0
    if hex[0] == '#fefefe':
        ind = 1
    rgb = hex_to_rgb(hex[ind])
    m = 10**5
    for i in range(len(COLORS)):
        cc = color_dict[COLORS[i]]
        d = dist(cc,rgb)
        dist_list.append(d)
    c_ind = 0
    min_value = min(dist_list)
    for j in range(len(dist_list)):
        if dist_list[j] == min_value:
            c_ind = j
    
    if c_ind == 0:
        card_color = [1,27]
    elif c_ind == 1:
        card_color = [28,54]
    elif c_ind == 2:
        card_color = [55,81]
    print(COLORS[c_ind])
    return card_color


#print(color_dist(get_colors(get_image('tempimgs/Slice11.png'), 5)))

def color_checker(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hex_list = get_colors(img, 4)
    out = color_dist(hex_list)
    
    return out

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

def list_compare(lst1, lst2):
    lst1_np = np.array(lst1)
    lst2_np = np.array(lst2)
    if all(lst1_np == lst2_np):
        return True
    else:
        return False

def color_factor(img):
    img1 = img.copy()
    img1 = img1[25 : 150, 35 : 265]
    img_shape = img1.shape
    x_coord, y_coord = random.randint(0,img_shape[1]), random.randint(0,img_shape[0])
    pix_color = color(img, x_coord, y_coord)
    pix_color = pix_color.tolist()
    while not(list_compare(pix_color,green)) or not(list_compare(pix_color, red)) or not(list_compare(pix_color, purple)):
        x_coord, y_coord = random.randint(0,img_shape[1]), random.randint(0,img_shape[0])
        pix_color = color(img, x_coord, y_coord)
    
    if list_compare(pix_color, green):
        card_color = [1,27]

    elif list_compare(pix_color, red):
        card_color = [28,54]

    elif list_compare(pix_color, purple):
        card_color = [55,81]
    return card_color
 
def card_to_vect():
    card_ind_list = []
    for i in range(1, card_num + 1):
        file_name = 'tempimgs/' + 'Slice'+str(i)+'.png'
        print(file_name)
        curr_img = get_image(file_name)
        ind_list = color_dist(get_colors(curr_img,5))
        start_ind, end_ind = ind_list[0], ind_list[1]
        curr_img = cv.resize(curr_img, (300,175))
        max_val, max_ind = 0, 0
        for j in range(start_ind, end_ind+1):
            curr_card_name = 'allcards/' + 'card' + number_buffer(j) + '.PNG'
            print(curr_card_name)
            curr_card = cv.imread(curr_card_name)
            curr_card = cv.resize(curr_card, (300,175))
            curr_card_trim = curr_card[25 : 150, 35 : 265]
            curr_hc = template_match2(curr_img, curr_card_trim) * template_match(curr_img, curr_card_trim) 
            if max(max_val, curr_hc) == curr_hc:
                max_val = max(max_val, curr_hc)
                max_ind = j
        #card_vect_list.append(int_to_list(ternary(max_ind)))
        card_ind_list.append(max_ind)
    
    return card_ind_list



img3_ans_list = [44,71,75,14,23,46,57,7,31,11,45,5]
splitter(img)
card_dict = make_card_dict()

card_list = card_to_vect()
card_vect_list = []
for i in range(len(card_list)):
    card_vect_list.append(card_dict[card_list[i]])

def check_set(arr1,arr2,arr3):
    ind_list_0 = [arr1[0], arr2[0], arr3[0]]
    ind_list_1 = [arr1[1], arr2[1], arr3[1]]
    ind_list_2 = [arr1[2], arr2[2], arr3[2]]
    ind_list_3 = [arr1[3], arr2[3], arr3[3]]

    if (len(set(ind_list_0)) == 1 or len(set(ind_list_0)) == 3) and (len(set(ind_list_1)) == 1 or len(set(ind_list_1)) == 3) and (len(set(ind_list_2)) == 1 or len(set(ind_list_2)) == 3) and (len(set(ind_list_3)) == 1 or len(set(ind_list_3)) == 3):
        return True
    else:
        return False