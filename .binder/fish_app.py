#from fastai2.vision.all import open_image, load_learner, image, torch
import torch
from fastai.vision.all import *
import streamlit as st
from PIL import Image
from pathlib import Path
import json



st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("FISH CLASSIFIER API")

introduction_str = 'This is a sea fish classifier.  It was made with 44,134 images corresponding to 44 species, which were scraped, \
    or downloaded manually in some cases. It was done by transfer of learning using the RESNET 34 model from Pytorch and the Fastai2 libraries. \
    In the validation of the model, the training error was 9%.'

st.markdown(introduction_str)


# Loading Model

@st.cache(persist=True)
fish_classifier = load_learner(Path(".",'fish_classification.pkl'))


# load Wikipedia dictionaries info
@st.cache(persist=True)
with open(Path(".",'name_dict.json')) as file1:
    name_dict= json.load(file1)
@st.cache(persist=True)    
with open(Path(".",'fish_summaries.json')) as file2:
    fish_summaries= json.load(file2)        
@st.cache(persist=True)    
with open(Path(".",'url_dict.json')) as file3:
    url_dict= json.load(file3)
    

# Loading File to classify

file_up = st.file_uploader(
    "Upload an image", 
    type=['png', 'jpg', 'jpeg'])


if file_up: 
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    #fish_classifier = torch.load('fish_classification.pkl')
    image = PILImage.create(file_up)
    pred,pred_idx,probs = fish_classifier.predict(image)

    st.subheader("Fish Prediction")
    out_label = f'Prediction: {name_dict[pred]};\n\n Probability: {probs[pred_idx]:.03f}'

    st.write(out_label)
    st.markdown("\n\n")
    st.subheader("Wikipedia Information About Fish Classificated")
    st.markdown("\n")
    st.write(fish_summaries[pred])
    st.markdown("\n")
    out_url = f"Wikipedia url source: {url_dict[pred]}"
    st.write(out_url)
 
