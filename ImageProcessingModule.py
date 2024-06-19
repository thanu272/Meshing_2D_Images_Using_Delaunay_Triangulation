#ImageProcessing Module
import sys
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from python_delaunay import Graph, Point, Edge, Triangle
import pygame
import base64
import random

width, height = 800, 600
# Function to convert PIL Image to NumPy array
def pil_to_np(image):
    return np.array(image)
def np_to_pil(image):
    return Image.fromarray(image)
# Function to adjust contrast of an image
def adjust_Contrast(image, factor):
    Contrast_enhancer = ImageEnhance.Brightness(image)
    Contrast_enhanced_image = Contrast_enhancer.enhance(factor)
    return Contrast_enhanced_image
# Function to adjust Brightness of an image
def adjust_Brightness(image, factor):
    Brightness_enhancer = ImageEnhance.Brightness(image)
    Brightness_enhanced_image = Brightness_enhancer.enhance(factor)
    return Brightness_enhanced_image

def adjust_RedChannel(image, factor):
    # Increase the intensity of the red channel
    red_channel = image[:,:,2]  # Extract the red channel
    red_channel = np.clip(red_channel + factor, 0, 255)  # Increase intensity by 50 (adjust as needed)
    image[:,:,2] = red_channel  # Update the red channel in the original image
    return image

def adjust_BlueChannel(image, factor):
    # Increase the intensity of the red channel
    blue_channel = image[:,:,0]  # Extract the blue channel
    blue_channel = np.clip(blue_channel - factor, 0, 255)  # Decrease intensity by 50 (adjust as needed)
    image[:,:,0] = blue_channel  # Update the blue channel in the original image
    return image

def adjust_GreenChannel(image, factor):
    # Increase the intensity of the red channel
    green_channel = image[:, :, 1]  # Extract the green channel
    green_channel = np.clip(green_channel + factor, 0, 255)  # Increase intensity by 50 (adjust as needed)
    image[:, :, 1] = green_channel  # Update the green channel in the original image
    return image

def Erosion(image):
    assert image is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(image,kernel,iterations = 1)
    return erosion

def Dilation(image):
    assert image is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(image,kernel,iterations = 1)
    return dilation

def Opening(image):
    assert image is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening

def Closing(image):
    assert image is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing

def Gradient(image):
    assert image is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5,5),np.uint8)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return gradient

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def load_image(uploaded_file):
	uploaded_img = Image.open(uploaded_file)
	return uploaded_img

def detect_Edge(myimage):
    #st.session_state.clicked = True
    Edge_enhancer = pil_to_np(myimage)
    Edge_enhanced_image = cv2.Canny(Edge_enhancer,100,200)
    Edge_enhanced_image = np_to_pil(Edge_enhanced_image)
    return Edge_enhanced_image

def bgremove1(myimage):
    myimage = pil_to_np(myimage)
    # Blur to image to reduce noise
    myimage = cv2.GaussianBlur(myimage,(5,5), 0)
 
    # We bin the pixels. Result will be a value 1..5
    bins=np.array([0,51,102,153,204,255])
    myimage[:,:,:] = np.digitize(myimage[:,:,:],bins,right=True)*51
 
    # Create single channel greyscale for thresholding
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
 
    # Perform Otsu thresholding and extract the background.
    # We use Binary Threshold as we want to create an all white background
    ret,background = cv2.threshold(myimage_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
 
    # Perform Otsu thresholding and extract the foreground.
    # We use TOZERO_INV as we want to keep some details of the foregorund
    ret,foreground = cv2.threshold(myimage_grey,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)  #Currently foreground is only a mask
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
 
    # Combine the background and foreground to obtain our final image
    finalimage = background+foreground
 
    return finalimage

def bgremove2(myimage):
    # First Convert to Grayscale
    myimage = pil_to_np(myimage)
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
 
    ret,baseline = cv2.threshold(myimage_grey,127,255,cv2.THRESH_TRUNC)
 
    ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)
 
    ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)
 
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
 
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
 
    # Combine the background and foreground to obtain our final image
    finalimage = background+foreground
    return finalimage

def bgremove3(myimage):
    # BG Remover 3
    myimage = pil_to_np(myimage)
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
    #Take S and remove any value that is less than half
    s = myimage_hsv[:,:,1]
    s = np.where(s < 127, 0, 1) # Any value below 127 will be excluded
 
    # We increase the brightness of the image and then mod by 255
    v = (myimage_hsv[:,:,2] + 127) % 255
    v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask
 
    # Combine our two masks based on S and V into a single "Foreground"
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
 
    background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
    foreground=cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
    finalimage = background+foreground # Combine foreground and background
 
    return finalimage

def bgremove4(myimage):
    # Read the image 
    src  = pil_to_np(myimage)
    
    # Convert image to image gray 
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 
    
    # Applying thresholding technique 
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY) 
    
    # Using cv2.split() to split channels  
    # of coloured image 
    b, g, r = cv2.split(src) 
    
    # Making list of Red, Green, Blue 
    # Channels and alpha 
    rgba = [b, g, r, alpha] 
    
    # Using cv2.merge() to merge rgba 
    # into a coloured/multi-channeled image 
    finalimage = cv2.merge(rgba, 4) 
     
    return finalimage

def BackgroundRemoval(image):
    #resizing the image
    # desired_width = 400  
    # aspect_ratio = image.shape[1] / image.shape[0]
    # desired_height = int(desired_width / aspect_ratio)
    # resized_image = cv2.resize(image, (desired_width, desired_height))
    # blk_thresh = 50
    #st.session_state.clicked = True
    enhancer = pil_to_np(image)
    
    # Read image
    #hh, ww = image.shape[:2]

    # threshold on white
    # Define lower and uppper limits
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(enhancer, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    result = cv2.bitwise_and(enhancer, enhancer, mask=mask)
        
    
    
    # # Convert image to image gray 
    # tmp = cv2.cvtColor(enhancer, cv2.COLOR_BGR2GRAY) 
    
    # # Applying thresholding technique 
    # _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY) 
    
    # # Using cv2.split() to split channels  
    # # of coloured image 
    # b, g, r = cv2.split(enhancer) 
    
    # # Making list of Red, Green, Blue 
    # # Channels and alpha 
    # rgba = [b, g, r, alpha] 
    
    # Using cv2.merge() to merge rgba 
    # into a coloured/multi-channeled image 
    #enhanced_image = cv2.merge(result, 4) 
    enhanced_image = np_to_pil(result)
    return enhanced_image

def detect_keypoints(image):
    # Use SIFT to detect keypoints
    enhancer = np.asarray(image)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(enhancer, None)

    # Extract the (x, y) coordinates of keypoints
    points = [Point(int(keypoint.pt[0]), int(keypoint.pt[1])) for keypoint in keypoints]
    return points

def draw_image_with_keypoints(image, keypoints):
    enhancer = np.asarray(image)
    for point in keypoints:
        cv2.circle(enhancer, (point._x, point._y), 3, (255, 255, 255), -1)
    return enhancer
    # cv2.imshow('Image with Keypoints', enhancer)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def create_triangular_mesh(keypoints):
    graph = Graph()
    print("Check")
        
    for point in keypoints:
        graph.addPoint(point)
    print("Check 1")
    graph.generateDelaunayMesh()
    print("DelaunayMesh Done")
    pygame.init()
    screen = pygame.display.set_mode([1024, 768])
    print("pygame.display Done")
    screen.fill((0, 0, 0))

    for p in graph._points:
        print(p)
        pygame.draw.circle(screen, (255, 255, 255), p.pos(), 3)

    for e in graph._edges:
        print(e)
        pygame.draw.line(screen, (0, 255, 0), e._a.pos(), e._b.pos())

    pygame.display.update()

    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.KEYDOWN:
                pygame.quit()
                sys.exit()    

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image

def generate_points(num_points):
    points = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_points)]
    return points

def draw_points(surface, points):
    for point in points:
        cv2.circle(surface, (255, 255, 255), point, 2)
        #pygame.draw.circle(surface, (255, 255, 255), point, 2)
    return surface

def create_mesh(points):
    # Your mesh generation logic here
    # This could involve Delaunay triangulation or a custom algorithm
    # For simplicity, let's draw lines connecting adjacent points
    mesh = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    return mesh

def draw_mesh(surface, mesh):
    for line in mesh:
        pygame.draw.line(surface, (255, 255, 255), line[0], line[1])

def ResizeImageResolution(image,width,height):
    res_0 = width * height
    res_1 = 250000

    # You need a scale factor to resize the image to res_1
    scale_factor = (res_1/res_0)**0.5
    resized = image.resize(width * scale_factor, height * scale_factor)
    return resized