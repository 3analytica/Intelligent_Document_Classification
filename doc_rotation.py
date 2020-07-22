import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def rotate_image(img):
    try:
        gray = img
        rot_data = pytesseract.image_to_osd(gray)
        rot = re.search('(?<=Rotate: )\d+', rot_data).group(0)
    
        angle = float(rot)
        if angle > 0:
            angle = 360 - angle
    
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except:
        pass
        

