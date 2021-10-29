import cv2
import numpy as np
import pyttsx3
import googletrans
from googletrans import Translator
import streamlit as st
import time

# This Function does transformation over the bounding boxes detected by the text detection model
def fourPointsTransform(frame, vertices):
    
    # Print vertices of each bounding box 
    vertices = np.asarray(vertices).astype(np.float32)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")
    
    # Apply perspective transform
    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result

# Defining function that returns the scale factor to resize the output image according to 
# the size/shape matrix of the input image
def setScaleFactor(frame):
    
    #Get image height and width
    frame_h, frame_w, ch = frame.shape
    
    # Define scaling factor according to preset values of the input image size so as 
    # the output image is scaled down to a generic size,"k" is the scaling factor value
    if (frame_h > 2000) and (frame_w > 2000):
        k = 0.2
    elif (2000 > frame_h > 1000) and 2000 > frame_w > 1000:
        k = 0.4
    elif (frame_h < 1000) or frame_w < 1000:
        k = 0.6
    elif (frame_h < 500) and frame_w < 500:
        k = 1
    else:
        k = 0.5
    return k


# Set title.
st.title('OpenCV Text-to-speach')

# Upload image.
uploaded_file = st.sidebar.file_uploader('Choose a text image', type='jpg')


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    inp_col, op_col = st.columns(2)
    with inp_col:
        st.header('Original')
        # Display uploaded image.
        orig = img.copy()
        st.image(orig, channels='BGR', use_column_width=True)

    # Define list to store the vocabulary in
    vocabulary =[]

    # Open file to import the vocabulary
    with open("/Users/aneerbanchakraborty/Documents/git/C0/c0-module-text-detection-ocr/resources/alphabet_94.txt") as f:

        # Read the file line by line
        for l in f:
        
            # Append each line into the vocabulary list.
            vocabulary.append(l.strip())
        
        #Close the file
        f.close()
    
    # DB model for text-detection based on resnet50
    text_detector = cv2.dnn_TextDetectionModel_DB("/Users/aneerbanchakraborty/Documents/git/C0/c0-module-text-detection-ocr/resources/DB_TD500_resnet50.onnx")
    binThresh = 0.3
    polyThresh = 0.5
    maxCandidates = 200
    unclipRatio = 2.0

    text_detector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh).setMaxCandidates(maxCandidates).setUnclipRatio(unclipRatio)
    text_detector.setInputParams(1.0/255, (736, 736), (122.67891434, 116.66876762, 104.00698793) , True)

    # CRNN model for text-recognition
    text_recogniser = cv2.dnn_TextRecognitionModel("/Users/aneerbanchakraborty/Documents/git/C0/c0-module-text-detection-ocr/resources/crnn_cs.onnx")
    text_recogniser.setDecodeType("CTC-greedy")
    text_recogniser.setVocabulary(vocabulary)
    text_recogniser.setInputParams(1/127.5, (100,32), (127.5, 127.5, 127.5))

    # Get the scaling factor
    scaling_factor = setScaleFactor(img)
    # Resize the image depending on the original image shape and the scaling factor
    image = cv2.resize(img, None, fx= scaling_factor, fy= scaling_factor, interpolation= cv2.INTER_LINEAR)

    # Use the DB text detector initialised previously to detect the presence of text in the image
    boxes, confs = text_detector.detect(image)

    text_data=[]

    #Iterate throught the bounding boxes detected by the text detector model
    for box in boxes:
        
        # Apply transformation on the bounding box detected by the text detection algorithm
        cropped_roi  = fourPointsTransform(image,box)
        
        # Recognise the text using the crnn model
        recResult = text_recogniser.recognize(cropped_roi)
        
        # Append recognised text to the data storage variable
        text_data.append(recResult)

    # Joining the text data together to form a output sentence
    text_data=' '.join(text_data)

    # Draw the bounding boxes of text detected.
    cv2.polylines(image, boxes, True, (255, 0, 255), 4)
    with op_col:
        st.header('Detected')
        st.image(image[:,:,::-1])

    translator = Translator()
    speechEngine =  pyttsx3.init()
    speechEngine. setProperty("rate", 120)
    st.sidebar.header('Detected Language: '+ googletrans.LANGUAGES[translator.detect(text_data).lang])
    list_of_lang = ['None']
    list_of_lang.extend(googletrans.LANGUAGES.values())
    lang = st.sidebar.selectbox('Translate into :', list_of_lang)

    if lang != 'None':
        text = translator.translate(text_data, lang).text
        
        st.title(text)
        
        # Speak out the translated text using the text to speech engine
        press = st.button('Speak')
        if press: 
            speechEngine.say(translator.translate(text_data, lang).text)
            speechEngine.runAndWait()
            time.sleep(7)
            speechEngine.endLoop()

    



