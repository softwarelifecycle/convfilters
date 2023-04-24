from timer import Timer
import cv2 as cv
import numpy as np
from multiprocessing import Pool

img = cv.imread("A-10.jpg")
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edgeKernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

def convolution(img, kernel):
    filtered_img = np.zeros_like(img)
    row, col = img.shape
    k_row, k_col = kernel.shape
    pad_height = k_row // 2
    pad_width = k_col // 2
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
    for i in range(row):
        for j in range(col):
            filtered_img[i, j] = np.sum(padded_img[i:i+k_row, j:j+k_col] * kernel)
    return filtered_img

def process_chunk(chunk):
    return convolution(chunk, edgeKernel)

def main(): 
    t = Timer()
    t.start()
    print("Starting Convolution!")
    pool = Pool()
    chunk_size = gray_img.shape[0] // 10
    chunks = [gray_img[i:i+chunk_size,:] for i in range(0, gray_img.shape[0], chunk_size)]
    filtered_chunks = pool.map(process_chunk, chunks)
    filtered_img = np.vstack(filtered_chunks)
    cv.imshow("original Image", gray_img)
    cv.imshow("Filtered Image", filtered_img)
    print("Done processing images!")
    t.stop()
    cv.imwrite("A-10filtered.jpg", filtered_img)
    cv.waitKey(0)

if __name__ == '__main__':
    main()
