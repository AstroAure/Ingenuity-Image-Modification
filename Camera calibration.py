import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog as fd
import sys

def loadingImages(folderName): #Creates a window to select pictures and returns them

    def select_files():
        global filenames
        filetypes = ( ('Images', '*.png *.jpg *.jpeg'), ('All files', '*.*') )
        filenames = fd.askopenfilenames(initialdir = folderName, filetypes = filetypes)

        if len(filenames) != 0:
            root.destroy()

    def on_closing():
        root.destroy()
        sys.exit("No pictures selected")


    root = tk.Tk()
    root.geometry("500x100")
    root.title("Pictures selection")

    frame=tk.Frame(root)
    frame.pack()

    open_button = tk.Button(frame, text='Select pictures', command=select_files)
    open_button.pack(pady=10)
    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()

    print(str(len(filenames)) + " pictures loaded")

    IMAGES = [cv.imread(file) for file in filenames]

    return IMAGES

def batchSave(IMAGES, foldername):
    for i, img in enumerate(IMAGES):
        cv.imwrite(foldername + str(i+1) + ".png", img)
        print("Saving : " + str(i+1) + " / " + str(len(IMAGES)))


def intrisicMatrix(img, f, cx, cy):
    h, w = img.shape[:2]
    Cx, Cy = w/2, h/2
    M = np.array([[f, 0, Cx+cx],
                  [0, f, Cy+cy],
                  [0, 0,   1    ]], dtype=float)
    newM = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(M, coeffs, (w,h), np.eye(3), None, 0, (w,h))
    return M, newM

def undistortCrop(img, M, coeffs, newM):
    h, w = img.shape[:2]
    mapx, mapy = cv.fisheye.initUndistortRectifyMap(M, coeffs, np.eye(3), newM, (w,h), cv.CV_32FC1)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR, None, cv.BORDER_CONSTANT)
    return dst

def batchUndistort(IMAGES, f, cx, cy, coeffs):
    M, newM = intrisicMatrix(IMAGES[0], f, cx, cy)
    UNDISTORTED = []
    for i, img in enumerate(IMAGES):
        dst = undistortCrop(img, M, coeffs, newM)
        UNDISTORTED.append(dst)
        print("Undistorting : " + str(i+1) + " / " + str(len(IMAGES)))
    return UNDISTORTED


IMAGES = loadingImages("FOLDER OF THE RAW IMAGES")

"""#Ingenuity NavCam
f = 276.38
cx = 4.159
cy = 9.588
coeffs = (-0.0202845, 0.0195991, -0.00834605, 0.00101331)"""

#Ingenuity RTE
f = 1586.47
cx = -10.977
cy = 34.348
coeffs = (0.503459, 1.11475, -1.71505, 0.60795)

UNDISTORTED = batchUndistort(IMAGES, f, cx, cy, coeffs)

batchSave(UNDISTORTED, "FOLDER TO SAVE THE IMAGES")