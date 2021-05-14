import cv2




def  find_horizontal_indexes(img,offset): 
    h_pixel_count = img.shape[1]
    v_pixel_count = img.shape[0]
    x_initial = 10000
    x_final = 0
    for j in range(-20,21):   
        for i in range(h_pixel_count):
            if(img[(v_pixel_count//2)+j][i][0]>offset and img[(v_pixel_count //2)+j][i][2]>offset and img[(v_pixel_count//2)+j][i][1]>offset):
                if(x_initial>i):
                    x_initial = i
                break

        for k in range(h_pixel_count):
            zh=h_pixel_count-k-1
            if(img[(v_pixel_count//2)+j][zh][0]>offset and img[(v_pixel_count//2)+j][zh][2]>offset and img[(v_pixel_count//2)+j][zh][1]>offset):
                if(x_final<zh):
                    x_final = zh
                break
    return (x_initial,x_final)



def  find_vertical_indexes(img,offset): 
    h_pixel_count = img.shape[1]
    v_pixel_count = img.shape[0]
    y_initial = 1000000
    y_final = 0
    for j in range(-20,21):   
        for i in range(v_pixel_count):
            if(img[i][(h_pixel_count//2)+j][0]>offset and img[i][(h_pixel_count//2)+j][1]>offset and img[i][(h_pixel_count//2)+j][2]>offset):
                if(y_initial>i):
                    y_initial = i
                break

        for k in range(v_pixel_count):
            zv=v_pixel_count-k-1
            if(img[zv][(h_pixel_count//2)+j][0]>offset and img[zv][(h_pixel_count//2)+j][1]>offset and img[zv][(h_pixel_count//2)+j][2]>offset):
                if(y_final<zv):
                    y_final = zv
                break
    return (y_initial,y_final)



def crop_resize(img,size,offset):

    x1,x2 = find_horizontal_indexes(img,offset)
    y1,y2 = find_vertical_indexes(img,offset)
    imgc = img[y1:y2,x1:x2,:]
    imgr = cv2.resize(imgc,(size))
    return imgr


def grey_clahe(img,clipLimitVal,tileGridSizeVal):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    obj_clahe = cv2.createCLAHE(clipLimit=clipLimitVal, tileGridSize=(tileGridSizeVal))
    output = obj_clahe.apply(img_grey)
    return output




def maxmin(img,minVal,maxVal):
    newmin = minVal
    newmax = maxVal
    img = newmin+(img - img.min())*(newmax - newmin)/(img.max()-img.min())
    return img






