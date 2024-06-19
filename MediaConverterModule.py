#MediaConverter
import cv2
from PIL import Image
import streamlit as st



Frames = []  # display every 300 frames
FramesCaptions = []  # display every 300 frames
frame_skip = 10 # display every 300 frames
def convert_VideoToFrames(uploaded_video):
    if uploaded_video is not None: # run only when user uploads video
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk

        st.markdown(f"""
        ### Files
        - {vid}
        """,
        unsafe_allow_html=True) # display file name

        vidcap = cv2.VideoCapture(vid) # load video from disk
        cur_frame = 0
        success = True
        while success:
            success, frame = vidcap.read() # get next frame from video
            if cur_frame % frame_skip == 0: # only analyze every n=300 frames
                print('frame: {}'.format(cur_frame)) 
                pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image
                Frames.append(pil_img)
                FramesCaptions.append('frame: {}'.format(cur_frame))
                #st.image(pil_img)
            cur_frame += 1
    
    return Frames