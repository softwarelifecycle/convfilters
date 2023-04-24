from timer import Timer
import cv2 as cv
import numpy as np

img = cv.imread("BadAssAlex.jpg")
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edgeKernel = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]

def convolution(img, kernel):
    row = img.shape[0] - len(kernel) + 1
    col = img.shape[1] - len(kernel[0]) + 1
    filtered_img = np.zeros(shape=(row, col))
    
    for i in range(row):
        for j in range(col):
            current = img[i : i+len(kernel), j : j+len(kernel[0])]
            multiplication = np.sum(current * kernel)
            filtered_img[i,j] = multiplication
    return filtered_img

def main():
    t = Timer()
    t.start()
    print("Starting Convolution!")
    filtered_img = convolution(gray_img, edgeKernel)
    cv.imshow("original Image", gray_img)
    cv.imshow("Filtered Image", filtered_img)
    print("Done processing images!")
    t.stop()
    cv.imwrite("Alexfiltered.jpg", filtered_img)
    cv.waitKey()

if __name__ == '__main__':
    main()
                