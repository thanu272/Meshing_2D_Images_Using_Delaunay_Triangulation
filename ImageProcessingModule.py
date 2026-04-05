#ImageProcessing Module
import sys
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from python_delaunay import Graph, Point, Edge, Triangle
import pygame
import base64
import random
import os

width, height = 800, 600

# Validation Constants
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
MIN_IMAGE_DIMENSION = 10
MAX_IMAGE_DIMENSION = 10000
MIN_CONTRAST_BRIGHTNESS_FACTOR = -100
MAX_CONTRAST_BRIGHTNESS_FACTOR = 100
MIN_CHANNEL_ADJUSTMENT = -255
MAX_CHANNEL_ADJUSTMENT = 255
MIN_KEYPOINTS = 3
MAX_KEYPOINTS = 10000

# Helper validation functions
def validate_image(image, param_name="image"):
    """Validate that image is not None and is valid format"""
    if image is None:
        raise ValueError(f"{param_name} cannot be None")
    if not isinstance(image, (Image.Image, np.ndarray)):
        raise TypeError(f"{param_name} must be PIL Image or numpy array, got {type(image)}")
    return True

def validate_image_dimensions(image):
    """Validate image dimensions are within acceptable range"""
    if isinstance(image, Image.Image):
        width, height = image.size
    elif isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:
        raise TypeError("Image must be PIL Image or numpy array")
    
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ValueError(f"Image dimensions too small: {width}x{height}. Minimum: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(f"Image dimensions too large: {width}x{height}. Maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}")
    return True

def validate_factor(factor, min_val=MIN_CONTRAST_BRIGHTNESS_FACTOR, max_val=MAX_CONTRAST_BRIGHTNESS_FACTOR, param_name="factor"):
    """Validate adjustment factor is within range"""
    if not isinstance(factor, (int, float)):
        raise TypeError(f"{param_name} must be numeric, got {type(factor)}")
    if not (min_val <= factor <= max_val):
        raise ValueError(f"{param_name} {factor} out of range [{min_val}, {max_val}]")
    return True

def validate_file_path(file_path):
    """Validate file exists and has supported extension"""
    if not isinstance(file_path, str):
        raise TypeError(f"File path must be string, got {type(file_path)}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"Unsupported format: {ext}. Supported: {SUPPORTED_IMAGE_FORMATS}")
    return True
# Function to convert PIL Image to NumPy array
def pil_to_np(image):
    validate_image(image, "image")
    validate_image_dimensions(image)
    try:
        return np.array(image)
    except Exception as e:
        raise RuntimeError(f"Failed to convert PIL Image to numpy array: {str(e)}")

def np_to_pil(image):
    validate_image(image, "image")
    validate_image_dimensions(image)
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(image)}")
    try:
        # Ensure image values are in valid range [0, 255]
        if image.max() > 255 or image.min() < 0:
            image = np.clip(image, 0, 255)
        return Image.fromarray(image.astype(np.uint8))
    except Exception as e:
        raise RuntimeError(f"Failed to convert numpy array to PIL Image: {str(e)}")
# Function to adjust contrast of an image
def adjust_Contrast(image, factor):
    validate_image(image, "image")
    validate_image_dimensions(image)
    validate_factor(factor, 0, 5, "Contrast factor")  # Typically 0-5 range
    try:
        if not isinstance(image, Image.Image):
            image = np_to_pil(image)
        Contrast_enhancer = ImageEnhance.Brightness(image)
        Contrast_enhanced_image = Contrast_enhancer.enhance(factor)
        return Contrast_enhanced_image
    except Exception as e:
        raise RuntimeError(f"Failed to adjust contrast: {str(e)}")

# Function to adjust Brightness of an image
def adjust_Brightness(image, factor):
    validate_image(image, "image")
    validate_image_dimensions(image)
    validate_factor(factor, 0, 5, "Brightness factor")  # Typically 0-5 range
    try:
        if not isinstance(image, Image.Image):
            image = np_to_pil(image)
        Brightness_enhancer = ImageEnhance.Brightness(image)
        Brightness_enhanced_image = Brightness_enhancer.enhance(factor)
        return Brightness_enhanced_image
    except Exception as e:
        raise RuntimeError(f"Failed to adjust brightness: {str(e)}")

def adjust_RedChannel(image, factor):
    validate_image(image, "image")
    validate_image_dimensions(image)
    validate_factor(factor, MIN_CHANNEL_ADJUSTMENT, MAX_CHANNEL_ADJUSTMENT, "Red channel factor")
    try:
        if isinstance(image, Image.Image):
            image = pil_to_np(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Image must have at least 3 color channels")
        
        # Increase the intensity of the red channel
        red_channel = image[:,:,2].astype(float)  # Extract the red channel
        red_channel = np.clip(red_channel + factor, 0, 255)  # Adjust intensity
        image[:,:,2] = red_channel.astype(np.uint8)  # Update the red channel
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to adjust red channel: {str(e)}")

def adjust_BlueChannel(image, factor):
    validate_image(image, "image")
    validate_image_dimensions(image)
    validate_factor(factor, MIN_CHANNEL_ADJUSTMENT, MAX_CHANNEL_ADJUSTMENT, "Blue channel factor")
    try:
        if isinstance(image, Image.Image):
            image = pil_to_np(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Image must have at least 3 color channels")
        
        # Adjust the blue channel
        blue_channel = image[:,:,0].astype(float)  # Extract the blue channel
        blue_channel = np.clip(blue_channel - factor, 0, 255)  # Decrease intensity
        image[:,:,0] = blue_channel.astype(np.uint8)  # Update the blue channel
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to adjust blue channel: {str(e)}")

def adjust_GreenChannel(image, factor):
    validate_image(image, "image")
    validate_image_dimensions(image)
    validate_factor(factor, MIN_CHANNEL_ADJUSTMENT, MAX_CHANNEL_ADJUSTMENT, "Green channel factor")
    try:
        if isinstance(image, Image.Image):
            image = pil_to_np(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Image must have at least 3 color channels")
        
        # Adjust the green channel
        green_channel = image[:, :, 1].astype(float)  # Extract the green channel
        green_channel = np.clip(green_channel + factor, 0, 255)  # Increase intensity
        image[:, :, 1] = green_channel.astype(np.uint8)  # Update the green channel
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to adjust green channel: {str(e)}")

def Erosion(image):
    validate_image(image, "image")
    validate_image_dimensions(image)
    try:
        if isinstance(image, Image.Image):
            image = pil_to_np(image)
        if image is None:
            raise ValueError("file could not be read, check with os.path.exists()")
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(image, kernel, iterations=1)
        return erosion
    except Exception as e:
        raise RuntimeError(f"Erosion operation failed: {str(e)}")

def Dilation(image):
    validate_image(image, "image")
    validate_image_dimensions(image)
    try:
        if isinstance(image, Image.Image):
            image = pil_to_np(image)
        if image is None:
            raise ValueError("file could not be read, check with os.path.exists()")
        kernel = np.ones((5,5), np.uint8)
        dilation = cv2.dilate(image, kernel, iterations=1)
        return dilation
    except Exception as e:
        raise RuntimeError(f"Dilation operation failed: {str(e)}")

def Opening(image):
    validate_image(image, "image")
    validate_image_dimensions(image)
    try:
        if isinstance(image, Image.Image):
            image = pil_to_np(image)
        if image is None:
            raise ValueError("file could not be read, check with os.path.exists()")
        kernel = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return opening
    except Exception as e:
        raise RuntimeError(f"Opening operation failed: {str(e)}")

def Closing(image):
    validate_image(image, "image")
    validate_image_dimensions(image)
    try:
        if isinstance(image, Image.Image):
            image = pil_to_np(image)
        if image is None:
            raise ValueError("file could not be read, check with os.path.exists()")
        kernel = np.ones((5,5), np.uint8)
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return closing
    except Exception as e:
        raise RuntimeError(f"Closing operation failed: {str(e)}")

def Gradient(image):
    validate_image(image, "image")
    validate_image_dimensions(image)
    try:
        if isinstance(image, Image.Image):
            image = pil_to_np(image)
        if image is None:
            raise ValueError("file could not be read, check with os.path.exists()")
        kernel = np.ones((5,5), np.uint8)
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        return gradient
    except Exception as e:
        raise RuntimeError(f"Gradient operation failed: {str(e)}")

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
    if uploaded_file is None:
        raise ValueError("uploaded_file cannot be None")
    
    try:
        # Check if file is a string path or file object
        if isinstance(uploaded_file, str):
            validate_file_path(uploaded_file)
            uploaded_img = Image.open(uploaded_file)
        else:
            # Assume it's a file-like object
            uploaded_img = Image.open(uploaded_file)
        
        # Validate loaded image
        validate_image(uploaded_img, "loaded image")
        validate_image_dimensions(uploaded_img)
        
        # Check color mode
        if uploaded_img.mode not in ['RGB', 'RGBA', 'L', 'P']:
            uploaded_img = uploaded_img.convert('RGB')
        
        return uploaded_img
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {str(e)}")

def detect_Edge(myimage):
    validate_image(myimage, "myimage")
    validate_image_dimensions(myimage)
    try:
        Edge_enhancer = pil_to_np(myimage)
        # Verify image has content
        if Edge_enhancer.size == 0:
            raise ValueError("Image is empty")
        
        # Convert to grayscale if needed
        if len(Edge_enhancer.shape) == 3:
            Edge_enhancer = cv2.cvtColor(Edge_enhancer, cv2.COLOR_BGR2GRAY)
        
        Edge_enhanced_image = cv2.Canny(Edge_enhancer, 100, 200)
        Edge_enhanced_image = np_to_pil(Edge_enhanced_image)
        return Edge_enhanced_image
    except Exception as e:
        raise RuntimeError(f"Edge detection failed: {str(e)}")

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
    """Detect keypoints using SIFT algorithm with validation"""
    validate_image(image, "image")
    validate_image_dimensions(image)
    
    try:
        enhancer = np.asarray(image)
        if enhancer.size == 0:
            raise ValueError("Image is empty")
        
        # Convert to grayscale if needed for SIFT
        if len(enhancer.shape) == 3 and enhancer.shape[2] == 3:
            enhancer = cv2.cvtColor(enhancer, cv2.COLOR_BGR2GRAY)
        elif len(enhancer.shape) == 3:
            enhancer = cv2.cvtColor(enhancer, cv2.COLOR_RGB2GRAY)
        
        sift = cv2.SIFT_create()
        keypoints = sift.detect(enhancer, None)
        
        # Validate keypoint count
        if len(keypoints) < MIN_KEYPOINTS:
            raise ValueError(f"Not enough keypoints detected: {len(keypoints)}. Minimum required: {MIN_KEYPOINTS}")
        if len(keypoints) > MAX_KEYPOINTS:
            print(f"Warning: Too many keypoints detected: {len(keypoints)}. Using first {MAX_KEYPOINTS}")
            keypoints = keypoints[:MAX_KEYPOINTS]
        
        # Extract the (x, y) coordinates of keypoints with validation
        points = []
        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            # Validate coordinates are within image bounds
            if 0 <= x < enhancer.shape[1] and 0 <= y < enhancer.shape[0]:
                points.append(Point(x, y))
        
        if len(points) == 0:
            raise ValueError("No valid keypoints found after coordinate validation")
        
        return points
    except Exception as e:
        raise RuntimeError(f"Keypoint detection failed: {str(e)}")

def draw_image_with_keypoints(image, keypoints):
    validate_image(image, "image")
    validate_image_dimensions(image)
    
    if not keypoints:
        raise ValueError("keypoints list cannot be empty")
    if not isinstance(keypoints, list):
        raise TypeError(f"keypoints must be a list, got {type(keypoints)}")
    
    try:
        enhancer = np.asarray(image).copy()
        
        for point in keypoints:
            if not hasattr(point, '_x') or not hasattr(point, '_y'):
                raise ValueError("Keypoint objects must have _x and _y attributes")
            
            # Validate coordinates are within image bounds
            if 0 <= point._x < enhancer.shape[1] and 0 <= point._y < enhancer.shape[0]:
                cv2.circle(enhancer, (point._x, point._y), 3, (255, 255, 255), -1)
        
        return enhancer
    except Exception as e:
        raise RuntimeError(f"Failed to draw keypoints: {str(e)}")

def create_triangular_mesh(keypoints):
    """Create triangular mesh from keypoints with validation"""
    if not keypoints:
        raise ValueError("keypoints list cannot be empty")
    if not isinstance(keypoints, list):
        raise TypeError(f"keypoints must be a list, got {type(keypoints)}")
    
    if len(keypoints) < MIN_KEYPOINTS:
        raise ValueError(f"Not enough keypoints for triangulation: {len(keypoints)}. Minimum required: {MIN_KEYPOINTS}")
    
    try:
        graph = Graph()
        print("Initializing graph...")
        
        # Add all keypoints to graph
        for point in keypoints:
            if not hasattr(point, '_x') or not hasattr(point, '_y'):
                raise ValueError("Keypoint objects must have _x and _y attributes")
            graph.addPoint(point)
        
        if len(graph._points) < MIN_KEYPOINTS:
            raise ValueError(f"Graph has insufficient points: {len(graph._points)}. Minimum required: {MIN_KEYPOINTS}")
        
        print(f"Added {len(graph._points)} points to graph")
        print("Generating Delaunay mesh...")
        
        graph.generateDelaunayMesh()
        
        # Validate mesh generation
        if not graph._triangles:
            raise RuntimeError("No triangles generated from Delaunay triangulation")
        if not graph._edges:
            raise RuntimeError("No edges generated from Delaunay triangulation")
        
        print(f"Generated {len(graph._triangles)} triangles and {len(graph._edges)} edges")
        
        # Initialize pygame display
        pygame.init()
        screen = pygame.display.set_mode([1024, 768])
        print("Pygame display initialized")
        screen.fill((0, 0, 0))

        # Draw points
        for p in graph._points:
            pygame.draw.circle(screen, (255, 255, 255), p.pos(), 3)

        # Draw edges
        for e in graph._edges:
            pygame.draw.line(screen, (0, 255, 0), e._a.pos(), e._b.pos())

        pygame.display.update()

        # Event loop
        while True:
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.KEYDOWN:
                    pygame.quit()
                    sys.exit()
    except Exception as e:
        if 'pygame' in str(e):
            print(f"Pygame error (non-critical): {str(e)}")
        else:
            raise RuntimeError(f"Failed to create triangular mesh: {str(e)}")    

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


def validate_image_processing(input_image, output_image=None, require_same_shape=True, require_color=False):
    """Validate image processing inputs and optional output.

    Parameters:
        input_image: PIL.Image.Image or numpy.ndarray
        output_image: optional processed image to validate
        require_same_shape: if True, output must match input dimensions
        require_color: if True, image must have 3 or 4 channels

    Returns:
        tuple(bool, str): (is_valid, error_message)
    """

    def _normalize_image(img):
        if img is None:
            return None, "image is None"
        if isinstance(img, Image.Image):
            arr = np.asarray(img)
        elif isinstance(img, np.ndarray):
            arr = img
        else:
            return None, f"unsupported image type: {type(img).__name__}"

        if arr.ndim not in (2, 3):
            return None, f"invalid image dimensions: expected 2D or 3D array, got {arr.ndim}D"
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            return None, "image width and height must be greater than zero"
        if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
            return None, f"invalid channel count: expected 1, 3, or 4 channels, got {arr.shape[2]}"
        if require_color and (arr.ndim != 3 or arr.shape[2] not in (3, 4)):
            return None, "image must have 3 or 4 color channels"
        if not np.issubdtype(arr.dtype, np.integer) and not np.issubdtype(arr.dtype, np.floating):
            return None, f"unsupported image dtype: {arr.dtype}"
        return arr, None

    input_arr, error = _normalize_image(input_image)
    if error:
        return False, f"invalid input image: {error}"

    if output_image is not None:
        output_arr, error = _normalize_image(output_image)
        if error:
            return False, f"invalid output image: {error}"
        if require_same_shape and input_arr.shape != output_arr.shape:
            return False, f"output image shape {output_arr.shape} does not match input image shape {input_arr.shape}"

    return True, ""


def ResizeImageResolution(image,width,height):
    res_0 = width * height
    res_1 = 250000

    # You need a scale factor to resize the image to res_1
    scale_factor = (res_1/res_0)**0.5
    resized = image.resize(width * scale_factor, height * scale_factor)
    return resized