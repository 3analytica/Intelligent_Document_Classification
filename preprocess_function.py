import cv2 

def preprocess(img):
    height, width = img.shape
    if height < 4000 and width < 3000:
        image = cv2.resize(img, None, fx = 2, fy = 2, interpolation=cv2.INTER_CUBIC)
    else:
        image = img
    blur = cv2.GaussianBlur(image,(5,5),0)
    ret3,image3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image3



