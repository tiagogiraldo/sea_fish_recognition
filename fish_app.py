#from fastai2.vision.all import open_image, load_learner, image, torch
import torch
from fastai.vision.all import *
import streamlit as st
from PIL import Image
from pathlib import Path
import json



st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("FISH SPECIES RECOGNITION API")

introduction_str = 'This is a sea fish classifier.  It was made with 44,134 images corresponding to 44 species, which were scraped, \
    or downloaded manually in some cases. It was done by transfer of learning using the RESNET 34 model from Pytorch and the Fastai2 libraries. \
    In the validation of the model, the training error was 7%.'

st.markdown(introduction_str)

print("Loading Model")
# Loading Model
#path = Path(".")
#@st.cache(persist=True)
fish_classifier = load_learner(Path(".",'fish_classification.pkl'))


# load Wikipedia dictionaries info
print("Loading names dictionary")
with open(Path(".",'name_dict.json')) as file1:
    name_dict= json.load(file1)
print("Loading summaries dictionary")    
with open(Path(".",'fish_summaries.json')) as file2:
    fish_summaries= json.load(file2)        
print("Loading url dictionary")    
with open(Path(".",'url_dict.json')) as file3:
    url_dict= json.load(file3)
    

# Loading File to classify
print("load image")
file_up = st.file_uploader(
    "Upload an image", 
    type=['png', 'jpg', 'jpeg'])


if file_up: 
    print('file was upload')
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    #fish_classifier = torch.load('fish_classification.pkl')
    image = PILImage.create(file_up)
    pred,pred_idx,probs = fish_classifier.predict(image)

    st.subheader("Fish Specie Prediction")
    out_label = f'Prediction: {name_dict[pred]};\n\n Probability: {probs[pred_idx]:.03f}'

    st.write(out_label)
    st.markdown("\n\n")
    st.subheader("Wikipedia Information About Classified Fish")
    st.markdown("\n")
    st.write(fish_summaries[pred])
    st.markdown("\n")
    out_url = f"Wikipedia url source: {url_dict[pred]}"
    st.write(out_url)
 


    # Wikipedia data



#from fastai2.vision.all import *
#from fastai2.vision.widgets import *



#inferencer = load_learner(path)

#img_bytes = st.file_uploader("Squash It!!", type=['png', 'jpg', 'jpeg'])
#if img_bytes is not None:
#    st.write("Image Uploaded Successfully:")
#    img = PIL.Image.open(img_bytes)

#    pred_class, pred_idx, outputs = inferencer.predict(img)
#    for out in outputs:
#        st.write(out)

#    st.write("Decision: ", pred_class)
