import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def create_img(size=(600, 600), text="Noise"):
    img = np.zeros(size, dtype=np.uint8)
    
    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 5, 25)[0]
    org = (size[0] // 2 - text_size[0] // 2, size[1] // 2 + text_size[1] // 2) #center in x and y
    
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 15)
    
    return img

def plot_img(img, size = (6, 8)):
    plt.figure(figsize = size)
    plt.imshow(img, cmap = plt.cm.gray)
    plt.axis("off")
    plt.show()