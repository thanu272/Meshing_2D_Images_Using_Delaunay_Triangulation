import streamlit as st
import cv2
from PIL import Image
from pathlib import Path
from streamlit_image_select import image_select
from ImageProcessingModule import *
import os

current_directory = Path.cwd()
# images_folder = current_directory / 'D:/PythonProjects/MidSemProjects/Images'
images_folder = current_directory / "D:/Python Projects/SixSemPeojects/ImageProcessing/Images"
Frames = []
FramesCaptions = []
Contrast = 0 
Brightness = 0
RedColor = 0
GreenColor = 0
BlueColor = 0
OriginalImage = Image.new("RGB", (100, 100), "white")
ProcessedImage = Image.new("RGB", (100, 100), "white")
st.session_state.Contrast_value = 0
st.session_state.Brightness_value = 0
st.session_state.RedColor_value = 0
st.session_state.GreenColor_value = 0
st.session_state.BlueColor_value = 0
st.session_state.MyImage = Image.new("RGB", (100, 100), "white")
ImageProcessed = False;         
# CachedImage = "D:/PythonProjects/MidSemProjects/Images/CachedImage.png"
CachedImage = "D:/Python Projects/SixSemPeojects/ImageProcessing/Images/CachedImage.png"
    
st.set_page_config(layout="wide")
st.write("""<style> 
                .st-emotion-cache-z5fcl4 {
                    width: 100%;
                    padding: 30px 10px 10px 10px;
                    min-width: auto;
                    max-width: initial;
                }
                .st-emotion-cache-hc3laj {
                    display: inline-flex;
                    -webkit-box-align: center;
                    align-items: center;
                    -webkit-box-pack: center;
                    justify-content: center;
                    font-weight: 400;
                    padding: 0.25rem 0.75rem;
                    border-radius: 0.5rem;
                    min-height: 38.4px;
                    margin: 0px;
                    line-height: 1.6;
                    color: green !important;
                    width: auto;
                    user-select: none;
                    background-color: rgb(43, 44, 54);
                    border: 1px solid rgba(250, 250, 250, 0.2);
                }
                .st-emotion-cache-q8sbsg {
                        font-family: "Source Sans Pro", sans-serif;
                    }
                .st-emotion-cache-16txtl3 {
                    padding: 50px 10px 10px 10px;
                }
                .scrollable-container {
                    overflow-y: scroll;
                    max-height: 300px; /* Set the maximum height as needed */
                }
                .container {
                    width: 100%;
                    margin: 0 auto;
                    overflow: hidden;
                }

                .left-column {
                    width: 200px; /* Static width for the left column */
                    float: left;
                    background-color: Black;
                    padding: 20px;
                    background-image: url('data:image/png;base64,{encoded_image}');
                }

                .content {
                    width: calc(100% - 600px); /* Adjusted width to account for left and right columns */
                    float: left;
                    background-color: #ffffff;
                    padding: 20px;
                }

                .right-column {
                    width: 400px; /* Static width for the right column */
                    float: left;
                    background-color: #ffffff;
                    padding: 20px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }

                th {
                    border: 1px solid #dddddd;
                    padding: 8px;
                    text-align: left;
                    width:50px;
                    color: #a4123f;
                }
                td {
                    border: 1px solid #dddddd;
                    padding: 8px;
                    text-align: left;
                    color: #a4123f;
                }
                h6 {
                    text-align: center;
                    color: #a4123f;
                }
            </Style>""",unsafe_allow_html=True)

try:
    AmritHeaderLogo = images_folder / 'AMRITLogo.png'
    AmritHeader = images_folder / 'AMRITHeader.png'
    AmrithaLogo = image_to_base64("D:/Python Projects/SixSemPeojects/ImageProcessing/Images/"+"AMRITLogo.png")
    AmrithaBanner = image_to_base64("D:/Python Projects/SixSemPeojects/ImageProcessing/Images/"+"AMRITHeader.png")
    st.markdown(f'''
                <div class="container">
                    <div class="left-column">
                        <!-- Left Column Content Goes Here -->
                        <img src="data:image/png;base64,'''+AmrithaLogo+'''" alt="Base64 Image" style="width: 100%;">
                    </div>
                    <div class="content">
                        <!-- Main Content Goes Here -->
                        <img src="data:image/png;base64,'''+AmrithaBanner+'''" alt="Base64 Image" style="height: 150px;display: block;margin: 0 auto;">
                    </div>
                    <div class="right-column">
                        <table>
                            <tr>
                                <td colspan="2"><h6>Project Description</h6></td>
                            </tr>
                            <tr>
                                <th>Project</th>
                                <td>Image Processing</td>
                            </tr>
                            <tr>
                                <th>Subject</th>
                                <td>Image mesh generation</td>
                            </tr>
                            <tr>
                                <th>Title</th>
                                <td>Titlr</td>
                            </tr>
                            <tr>
                                <th>Participants</th>
                                <td>Thanushri Madhuraj <br> Another member</td>
                            </tr>
                        </table>
                    </div>
                </div>
                 ''', unsafe_allow_html=True)
except FileNotFoundError:
    st.error(f"Image 'Headliner.png' not found in '{images_folder}'. Please check the file path.") 
except Exception as e:
    st.error(f"An error occurred: {e}")
    
# Create tabs
with st.sidebar:
    
    with st.expander('Select a Video File', expanded=True):
        uploaded_video = st.file_uploader("", type=["mp4", "mov"])
        frame_skip = 10 # display every 300 frames

    with st.expander('Select a Image File', expanded=False):
        uploaded_file = st.file_uploader("", type="jpg")
        frame_skip = 10 # display every 300 frames


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
            
    if uploaded_file is not None:
        Frames.append(load_image(uploaded_file))
        FramesCaptions.append(uploaded_file.name)
        
    if Frames:
        OriginalImage = image_select(
            label="Select an Image",
            images=Frames,
            captions=FramesCaptions,
            )
        cv2.imwrite(CachedImage, pil_to_np(OriginalImage))
        
    
Previewer,Functions = st.columns([3,1])
with Previewer:
    # Display the original image
    Contrast = 1 
    original_image_placeholder = st.empty()
    if os.path.exists(CachedImage):
        src = cv2.imread(CachedImage, 1)
        original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
    # if OriginalImage:
    #     if ProcessedImage:
    #         original_image_placeholder.image(ProcessedImage, caption="Original Image", use_column_width=True)
    #     else:
    #         original_image_placeholder.image(OriginalImage, caption="Processed Image", use_column_width=True)
        
    #st.image(adjust_contrast(img, Contrast), caption='Educative')
    
with Functions:
    if os.path.exists(CachedImage):
        src = cv2.imread(CachedImage, 1)
        original_image_placeholder.image(src, caption="Original Image", use_column_width=True)

    tab1, tab2, tab3 = st.tabs(["Adjust Image", "Morphology", "Create Mesh"])
    with tab1:
        with st.container():
            Contrast = st.slider("Adjust Contrast",-127, 127, 0, key="Contrast_slider")
            if Contrast != st.session_state.Contrast_value:
                st.session_state.Contrast_value = Contrast
                if os.path.exists(CachedImage):
                    src = cv2.imread(CachedImage, 1)
                    src = apply_brightness_contrast(src,st.session_state.Contrast_value,st.session_state.Brightness_value)
                    cv2.imwrite(CachedImage, pil_to_np(src))
                    #ProcessedImage = adjust_Contrast(np_to_pil(src), Contrast)
                    original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
        with st.container():
            Brightness = st.slider("Adjust Brightness",-64, 67, 0, key="Brightness_slider")
            if Brightness != st.session_state.Brightness_value:
                st.session_state.Brightness_value = Brightness
                if os.path.exists(CachedImage):
                    src = cv2.imread(CachedImage, 1)
                    # ProcessedImage = adjust_Brightness(src, Contrast)
                    src = apply_brightness_contrast(src,st.session_state.Contrast_value,st.session_state.Brightness_value)
                    cv2.imwrite(CachedImage, pil_to_np(src))
                    original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
                    
        with st.container():
            RedColor = st.slider("Adjust adjust_RedChannel",-50, 50, 0, key="adjust_RedChannel_slider")
            if RedColor != st.session_state.RedColor_value:
                st.session_state.RedColor_value = RedColor
                if os.path.exists(CachedImage):
                    src = cv2.imread(CachedImage, 1)
                    # ProcessedImage = adjust_Brightness(src, Contrast)
                    src = adjust_RedChannel(src,st.session_state.RedColor_value)
                    #cv2.imwrite(CachedImage, pil_to_np(src))
                    original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
        with st.container():
            BlueColor = st.slider("Adjust adjust_BlueChannel",-50, 50, 0, key="adjust_BlueChannel_slider")
            if BlueColor != st.session_state.BlueColor_value:
                st.session_state.BlueColor_value = BlueColor
                if os.path.exists(CachedImage):
                    src = cv2.imread(CachedImage, 1)
                    # ProcessedImage = adjust_Brightness(src, Contrast)
                    src = adjust_BlueChannel(src,st.session_state.BlueColor_value)
                    #cv2.imwrite(CachedImage, pil_to_np(src))
                    original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
        with st.container():
            GreenColor = st.slider("Adjust adjust_GreenChannel",-50, 50, 0, key="adjust_GreenChannel_slider")
            if GreenColor != st.session_state.GreenColor_value:
                st.session_state.GreenColor_value = GreenColor
                if os.path.exists(CachedImage):
                    src = cv2.imread(CachedImage, 1)
                    # ProcessedImage = adjust_Brightness(src, Contrast)
                    src = adjust_GreenChannel(src,st.session_state.GreenColor_value)
                    #cv2.imwrite(CachedImage, pil_to_np(src))
                    original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
                    
    with tab2: 
        #Erosion & Dilation
        with st.container():
            Column01,Column02 = st.columns([1,1])
            with Column01:
                with st.form(key="Erosion"):
                    Erosionsubmit = st.form_submit_button("Erosion",type="secondary",use_container_width=True)
                    if Erosionsubmit:
                        if os.path.exists(CachedImage):
                            src = cv2.imread(CachedImage, 1)
                            src = Erosion(src)
                            cv2.imwrite(CachedImage, pil_to_np(src))
                            original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
            with Column02:
                with st.form(key="Dilation"):
                    Dilationsubmit = st.form_submit_button("Dilation",type="secondary",use_container_width=True)
                    if Dilationsubmit:
                        if os.path.exists(CachedImage):
                            src = cv2.imread(CachedImage, 1)
                            src = Dilation(src)
                            cv2.imwrite(CachedImage, pil_to_np(src))
                            original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
        #Opening & Closing
        with st.container():
            Column01,Column02 = st.columns([1,1])
            with Column01:
                with st.form(key="Opening"):
                    Openingsubmit = st.form_submit_button("Opening",type="secondary",use_container_width=True)
                    if Openingsubmit:
                        if os.path.exists(CachedImage):
                            src = cv2.imread(CachedImage, 1)
                            src = Opening(src)
                            cv2.imwrite(CachedImage, pil_to_np(src))
                            original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
            with Column02:
                with st.form(key="Closing"):
                    Closingsubmit = st.form_submit_button("Closing",type="secondary",use_container_width=True)
                    if Closingsubmit:
                        if os.path.exists(CachedImage):
                            src = cv2.imread(CachedImage, 1)
                            src = Closing(src)
                            cv2.imwrite(CachedImage, pil_to_np(src))
                            original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
        #Gradient
        with st.container():
            Column01,Column02 = st.columns([1,1])
            with Column01:
                with st.form(key="Gradient"):
                    submit = st.form_submit_button("Gradient",type="secondary",use_container_width=True)
                    if submit:
                        if os.path.exists(CachedImage):
                            src = cv2.imread(CachedImage, 1)
                            src = Gradient(src)
                            cv2.imwrite(CachedImage, pil_to_np(src))
                            original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
            with Column02:
                with st.form(key="Edge"):
                    submit = st.form_submit_button("Detect Edge",type="secondary",use_container_width=True)
                    if submit:
                        if os.path.exists(CachedImage):
                            src = cv2.imread(CachedImage, 1)
                            src = detect_Edge(src)
                            cv2.imwrite(CachedImage, pil_to_np(src))
                            original_image_placeholder.image(src, caption="Original Image", use_column_width=True)    
        with st.container():
            Column01,Column02 = st.columns([1,1])
            with Column01:
                with st.form(key="Background"):
                    submit = st.form_submit_button("Remove Background",type="secondary",use_container_width=True)
                    if submit:
                        if os.path.exists(CachedImage):
                            src = cv2.imread(CachedImage, 1)
                            src = bgremove3(src)
                            cv2.imwrite(CachedImage, pil_to_np(src))
                            original_image_placeholder.image(src, caption="Original Image", use_column_width=True)
        
    with tab3:     
        with st.container():
            Column01,Column02 = st.columns([1,1])
            with Column01:
                with st.form(key="ResizeImage"):
                    submit = st.form_submit_button("Resize Image",type="secondary",use_container_width=True)
                    if submit:
                        if os.path.exists(CachedImage):
                            src = cv2.imread(CachedImage, 1)
                            src = ResizeImageResolution(src,src.shape[1],src.shape[2])
                            original_image_placeholder.image(src, caption="Adjusted Image", use_column_width=True)                 
        with st.container():
            Column01,Column02 = st.columns([1,1])
            with Column01:
                with st.form(key="KeyPoints"):
                    submit = st.form_submit_button("Key Points",type="secondary",use_container_width=True)
                    if submit:
                        if os.path.exists(CachedImage):
                            src = cv2.imread(CachedImage, 1)
                            keypoints = detect_keypoints(src)
                            src = draw_image_with_keypoints(src, keypoints)
                            original_image_placeholder.image(src, caption="Adjusted Image", use_column_width=True)
            with Column02:
                with st.form(key="MeshGeneration"):
                    submit = st.form_submit_button("Generate Mesh",type="secondary",use_container_width=True)
                    if submit:
                        if os.path.exists(CachedImage):
                            src = cv2.imread(CachedImage, 1)
                            keypoints = detect_keypoints(src)
                            print("Key points Generated")
                            create_triangular_mesh(keypoints)
                            print("Triangular mesh Created")
                            # original_image_placeholder.image(src, caption="Adjusted Image", use_column_width=True) 