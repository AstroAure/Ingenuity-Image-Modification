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

def batchResizeScale(IMAGES, scale): #Resizes images according to a percentage
    RESIZED = []
    for i, img in enumerate(IMAGES):
        h, w = img.shape[:2]
        img = cv.resize(img, (int(w*scale), int(h*scale)))
        RESIZED.append(img)
        print("Resizing : " + str(i+1) + " / " + str(len(IMAGES)))
    return RESIZED


def batchResizeWidth(IMAGES, width): #Resizes images according to a width
    RESIZED = []
    for i, img in enumerate(IMAGES):
        h, w = img.shape[:2]
        img = cv.resize(img, (width, int(h*width/w)))
        RESIZED.append(img)
        print("Resizing : " + str(i+1) + " / " + str(len(IMAGES)))
    return RESIZED


def batchResizeHeight(IMAGES, height): #Resizes images according to a height
    RESIZED = []
    for i, img in enumerate(IMAGES):
        h, w = img.shape[:2]
        img = cv.resize(img, (int(w*height/h), height))
        RESIZED.append(img)
        print("Resizing : " + str(i+1) + " / " + str(len(IMAGES)))
    return RESIZED


def batchCvtColor(IMAGES, color): #Converts color space of images
    CONVERTED = []
    for i, img in enumerate(IMAGES):
        img = cv.cvtColor(img, color)
        CONVERTED.append(img)
        print("Converting color : " + str(i+1) + " / " + str(len(IMAGES)))
    return CONVERTED


def batchSave(IMAGES, foldername): #Saves many images
    for i, img in enumerate(IMAGES):
        cv.imwrite(foldername + str(i+1) + ".png", img)
        print("Saving : " + str(i+1) + " / " + str(len(IMAGES)))


def batchGammaCorrection(IMAGES, gamma): #Modify the color of the images with a gamma correction
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    GAMMA = []
    for i, img, in enumerate(IMAGES):
        res = cv.LUT(img, lookUpTable)
        GAMMA.append(res)
        print("Gamma correction : " + str(i+1) + " / " + str(len(IMAGES)))
    return GAMMA


def batchCrop(IMAGES, ROI): #Crops images according to a ROI
    CROPPED = []
    for i, img in enumerate(IMAGES):
        h, w = img.shape[:2]
        roi_top_left, roi_bottom_right = ROI
        top_left = (int(roi_top_left[0]*w), int(roi_top_left[1]*h))
        bottom_right = (int(roi_bottom_right[0]*w), int(roi_bottom_right[1]*h))
        dst = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        CROPPED.append(dst)
        print("Cropping : " + str(i+1) + " / " + str(len(IMAGES)))
    return CROPPED


def batchSharpen(IMAGES): #Sharpens images
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    SHARP = []
    for i, img, in enumerate(IMAGES):
        res = cv.filter2D(img, -1, filter)
        SHARP.append(res)
        print("Sharpening : " + str(i+1) + " / " + str(len(IMAGES)))
    return SHARP


def batchGaussianBlur(IMAGES, size): #Blurs images
    BLURRED = []
    for i, img, in enumerate(IMAGES):
        res = cv.GaussianBlur(img, (2*size+1, 2*size+1), 0)
        BLURRED.append(res)
        print("Blurring : " + str(i+1) + " / " + str(len(IMAGES)))
    return BLURRED


def batchDrawPoints(IMAGES, POINTS): #Draw points on images
    DRAWN = []
    for i, (img, points) in enumerate(zip(IMAGES, POINTS)):
        for point in points:
            dst = cv.circle(img, point.astype(int), 3, (255, 0, 255), -1)
        DRAWN.append(dst)
        print("Drawing features : " + str(i+1) + " / " + str(len(IMAGES)))
    return DRAWN


def batchVignetting(IMAGES, size, center, correction): #Corrects vignetting
    cx, cy = center
    VIGNETTED = []
    for i, img in enumerate(IMAGES):
        h, w = img.shape[:2]
        X_kernel = cv.getGaussianKernel(w+2*abs(cx), size)
        Y_kernel = cv.getGaussianKernel(h+2*abs(cy), size)
        kernel = Y_kernel * X_kernel.T
        mask = (kernel/np.max(kernel))[abs(cy)-cy:abs(cy)-cy+h, abs(cx)-cx:abs(cx)-cx+w]
        if correction:
            mask = 1-(mask-np.min(mask))
        dst = np.copy(img)
        if len(dst.shape)==3:
            for i in range(dst.shape[2]):
                dst[:,:,i] = dst[:,:,i] * mask
        else: dst = dst * mask
        dst = np.minimum(dst, 250)
        dst = dst.astype(np.uint8)
        VIGNETTED.append(dst)
        print("Vignetting : " + str(i+1) + " / " + str(len(IMAGES)))
    return VIGNETTED


def batchBrigthness(IMAGES, alpha, beta): #Modifies brightness
    BRIGHT = []
    for i, img in enumerate(IMAGES):
        dst = np.copy(img)
        if len(dst.shape)==3:
            for i in range(dst.shape[2]):
                dst[:,:,i] = dst[:,:,i] * alpha + beta
        else: dst = dst * alpha + beta
        dst = np.minimum(dst, 250)
        dst = dst.astype(np.uint8)
        BRIGHT.append(dst)
        print("Brightening : " + str(i+1) + " / " + str(len(IMAGES)))
    return BRIGHT


def intrisicMatrix(img, f, cx, cy): #Defines the intrisic matrix to undistort images
    h, w = img.shape[:2]
    Cx, Cy = w/2, h/2
    M = np.array([[f, 0, Cx+cx],
                  [0, f, Cy+cy],
                  [0, 0,   1    ]], dtype=float)
    newM = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(M, coeffs, (w,h), np.eye(3), None, 0, (w,h))
    return M, newM


def undistortCrop(img, M, coeffs, newM): #Undistorts one image
    h, w = img.shape[:2]
    mapx, mapy = cv.fisheye.initUndistortRectifyMap(M, coeffs, np.eye(3), newM, (w,h), cv.CV_32FC1)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR, None, cv.BORDER_CONSTANT)
    return dst


def batchUndistort(IMAGES, f, cx, cy, coeffs): #Undistorts many images
    M, newM = intrisicMatrix(IMAGES[0], f, cx, cy)
    UNDISTORTED = []
    for i, img in enumerate(IMAGES):
        dst = undistortCrop(img, M, coeffs, newM)
        UNDISTORTED.append(dst)
        print("Undistorting : " + str(i) + " / " + str(len(IMAGES)))
    return UNDISTORTED


def batchHistEqualization(IMAGES): #Equalizes the brightness histogram of images
    EQUALIZED = []
    for i, img, in enumerate(IMAGES):
        res = cv.equalizeHist(img)
        EQUALIZED.append(res)
        print("Equalizing : " + str(i+1) + " / " + str(len(IMAGES)))
    return EQUALIZED


def batchMask(IMAGES, mask): #Applies a mask to many images
    MASKED = []
    for i, img in enumerate(IMAGES):
        dst = np.copy(img)
        if len(dst.shape)==3:
            for i in range(dst.shape[2]):
                dst[:,:,i] = dst[:,:,i] * mask
        else:
            dst = dst * mask
        dst = np.minimum(dst, 250)
        dst = dst.astype(np.uint8)
        MASKED.append(dst)
        print("Masking : " + str(i+1) + " / " + str(len(IMAGES)))
    return MASKED

def vignettingMask(shape, size, center, sigma, correction): #Creates a vignetting mask
    h, w = shape
    cx, cy = center
    sx, sy = sigma
    X_kernel = cv.getGaussianKernel(w+2*abs(cx), size*sx)
    Y_kernel = cv.getGaussianKernel(h+2*abs(cy), size*sy)
    kernel = Y_kernel * X_kernel.T
    mask = (kernel/np.max(kernel))[abs(cy)-cy:abs(cy)-cy+h, abs(cx)-cx:abs(cx)-cx+w]
    if correction:
        mask = 1-(mask-np.min(mask))
    return mask

def gradientMask(shape, axis, min, max, orientation): #Defines a gradient mask
    l = shape[axis]
    mask = np.zeros(shape)
    for i in range(l):
        if axis == 0:
            if orientation == 1:
                mask[i,:] = min + i*(max-min)/l
            elif orientation == -1:
                mask[i,:] = max + i*(min-max)/l
        elif axis == 1:
            if orientation == 1:
                mask[:,i] = min + i*(max-min)/l
            elif orientation == -1:
                mask[:,i] = max + i*(min-max)/l
    return mask


def batchThreshold(IMAGES, threshold, maxval, type): #Creates B&W images according to a threshold
    THRESH = []
    for i, img, in enumerate(IMAGES):
        retval, res = cv.threshold(img, threshold, maxval, type)
        THRESH.append(res)
        print("Thresholding : " + str(i+1) + " / " + str(len(IMAGES)))
    return THRESH


def batchSigmoid(IMAGES, center, sigma): #Modifies the brightness of images according to a sigmoid curve
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(1/(1+np.exp((center-i)*sigma)) * 255.0, 0, 255)
    #plt.plot(range(256), lookUpTable[0]), plt.show()

    SIGMOID = []
    for i, img, in enumerate(IMAGES):
        res = cv.LUT(img, lookUpTable)
        SIGMOID.append(res)
        print("Sigmoid correction : " + str(i+1) + " / " + str(len(IMAGES)))
    return SIGMOID



IMAGES = loadingImages("FOLDER OF THE RAW IMAGES")

#Ingenuity NavCam
f = 276.38
cx = 4.159
cy = 9.588
coeffs = (-0.0202845, 0.0195991, -0.00834605, 0.00101331)

UNDISTORTED = batchUndistort(IMAGES, f, cx, cy , coeffs)
CONVERTED = batchCvtColor(UNDISTORTED, cv.COLOR_BGR2GRAY)
RESIZED = batchResizeHeight(CONVERTED, 720)
GRADIENT = batchMask(RESIZED, gradientMask((RESIZED[0].shape[:2]), 0, 0.65, 1, -1))
CENTER = batchMask(GRADIENT, cv.addWeighted(gradientMask((RESIZED[0].shape[:2]), 1, 0.5, 1, -1), 0.5 , gradientMask((RESIZED[0].shape[:2]), 1, 0.5, 1, 1), 0.5, 0))
VIGNETTED = batchMask(CENTER, vignettingMask((RESIZED[0].shape[:2]), 500, (0,30), (1.5, 1), 1))
SIGMOID = batchSigmoid(VIGNETTED, 80, 0.1)

#RESIZED = batchResizeHeight(IMAGES, 1080)

batchSave(SIGMOID, "FOLDER OF THE MODIFIED IMAGES")


"""plt.subplot(2,3,1), plt.imshow(RESIZED[10], cmap='gray')
plt.subplot(2,3,2), plt.imshow(GRADIENT[10], cmap='gray')
plt.subplot(2,3,3), plt.imshow(CENTER[10], cmap='gray')
plt.subplot(2,3,4), plt.imshow(VIGNETTED[10], cmap='gray')
plt.subplot(2,3,6), plt.imshow(SIGMOID[10], cmap='gray')

plt.show()"""