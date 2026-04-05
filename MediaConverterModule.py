#MediaConverter
import cv2
from PIL import Image
import streamlit as st
import os

# Validation Constants
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
MIN_FRAME_SKIP = 1
MAX_FRAME_SKIP = 1000
MIN_VIDEO_DURATION = 0.1  # seconds
MAX_VIDEO_SIZE_MB = 500  # MB

Frames = []  # display every 300 frames
FramesCaptions = []  # display every 300 frames
frame_skip = 10 # display every 300 frames

def validate_video_file(file_path):
    """Validate video file exists and has supported extension"""
    if not isinstance(file_path, str):
        raise TypeError(f"File path must be string, got {type(file_path)}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    # Check file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_FORMATS:
        raise ValueError(f"Unsupported video format: {ext}. Supported: {SUPPORTED_VIDEO_FORMATS}")
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_VIDEO_SIZE_MB:
        raise ValueError(f"Video file too large: {file_size_mb:.2f}MB. Maximum: {MAX_VIDEO_SIZE_MB}MB")
    
    if file_size_mb < 0.01:
        raise ValueError("Video file is too small or empty")
    
    return True

def validate_frame_skip(frame_skip):
    """Validate frame skip parameter"""
    if not isinstance(frame_skip, int):
        raise TypeError(f"frame_skip must be integer, got {type(frame_skip)}")
    
    if not (MIN_FRAME_SKIP <= frame_skip <= MAX_FRAME_SKIP):
        raise ValueError(f"frame_skip {frame_skip} out of range [{MIN_FRAME_SKIP}, {MAX_FRAME_SKIP}]")
    
    return True

def validate_video_object(video_object, object_name="video_object"):
    """Validate video object properties"""
    if video_object is None:
        raise ValueError(f"{object_name} cannot be None")
    
    # Check if VideoCapture opened successfully
    if not video_object.isOpened():
        raise RuntimeError(f"Failed to open video: {object_name}")
    
    # Get video properties
    fps = video_object.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_object.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_object.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")
    
    if total_frames <= 0:
        raise ValueError(f"Invalid total frames: {total_frames}")
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame dimensions: {width}x{height}")
    
    duration = total_frames / fps
    if duration < MIN_VIDEO_DURATION:
        raise ValueError(f"Video duration too short: {duration:.2f}s. Minimum: {MIN_VIDEO_DURATION}s")
    
    return {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration': duration
    }

def validate_frame(frame, frame_number):
    """Validate individual video frame"""
    if frame is None:
        raise ValueError(f"Failed to read frame {frame_number}")
    
    if not isinstance(frame, type(Image.new('RGB', (1, 1)))):
        # For numpy array
        if len(frame.shape) < 2:
            raise ValueError(f"Frame {frame_number} has invalid shape: {frame.shape}")
    
    return True
def convert_VideoToFrames(uploaded_video, skip_frames=None):
    """
    Convert video to frames with comprehensive validation
    
    Parameters:
        uploaded_video: Uploaded video object with .name and .read() methods
        skip_frames: Optional frame skip parameter (default: use module-level frame_skip)
    
    Returns:
        list: Extracted frames as PIL Images
    """
    global Frames, FramesCaptions, frame_skip
    
    # Validate input
    if uploaded_video is None:
        raise ValueError("uploaded_video cannot be None")
    
    if not hasattr(uploaded_video, 'name') or not hasattr(uploaded_video, 'read'):
        raise AttributeError("uploaded_video must have 'name' and 'read' attributes")
    
    # Reset frame lists
    Frames = []
    FramesCaptions = []
    
    try:
        vid_name = str(uploaded_video.name)
        if not vid_name:
            raise ValueError("Video file name is empty")
        
        # Save video to disk
        with open(vid_name, mode='wb') as f:
            video_data = uploaded_video.read()
            if not video_data:
                raise ValueError("Video file is empty or unreadable")
            f.write(video_data)
        
        # Validate saved file
        validate_video_file(vid_name)
        
        # Display file information
        st.markdown(f"""
        ### Files
        - {vid_name}
        """, unsafe_allow_html=True)
        
        # Open video
        vidcap = cv2.VideoCapture(vid_name)
        
        # Validate video object
        video_properties = validate_video_object(vidcap, vid_name)
        print(f"Video properties: {video_properties}")
        
        # Determine frame skip value
        current_skip = skip_frames if skip_frames is not None else frame_skip
        validate_frame_skip(current_skip)
        
        # Extract frames
        cur_frame = 0
        success = True
        frames_extracted = 0
        
        while success:
            success, frame = vidcap.read()
            
            if not success:
                break
            
            # Validate frame
            try:
                validate_frame(frame, cur_frame)
            except Exception as e:
                print(f"Warning: Skipping frame {cur_frame}: {str(e)}")
                cur_frame += 1
                continue
            
            # Process every nth frame
            if cur_frame % current_skip == 0:
                try:
                    # Convert to PIL Image
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # Validate converted image
                    if pil_img.size[0] <= 0 or pil_img.size[1] <= 0:
                        raise ValueError(f"Invalid image size: {pil_img.size}")
                    
                    Frames.append(pil_img)
                    FramesCaptions.append(f'frame: {cur_frame}')
                    frames_extracted += 1
                    print(f'Extracted frame: {cur_frame}')
                    
                except Exception as e:
                    print(f"Warning: Failed to extract frame {cur_frame}: {str(e)}")
            
            cur_frame += 1
        
        # Validate frame extraction
        if frames_extracted == 0:
            raise RuntimeError("No frames were successfully extracted from video")
        
        print(f"Successfully extracted {frames_extracted} frames")
        
        vidcap.release()
        
        # Cleanup: Remove temporary video file
        try:
            os.remove(vid_name)
        except Exception as e:
            print(f"Warning: Could not delete temporary video file {vid_name}: {str(e)}")
        
        return Frames
    
    except Exception as e:
        # Ensure video file is closed on error
        try:
            if 'vidcap' in locals():
                vidcap.release()
        except:
            pass
        
        raise RuntimeError(f"Video conversion failed: {str(e)}")