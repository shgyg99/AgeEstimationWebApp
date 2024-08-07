import streamlit as st
import os
os.system('pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git')
import numpy as np
import time
import cv2
import face_detection
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torchvision.models import resnet50, ResNet50_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = 'model_resnet50_loss=3.7.pt'

def picture_upload(model_path, image, transform):    
    img_array = np.frombuffer(image, dtype=np.uint8)
    image_ = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()

    detector = face_detection.build_detector(
            "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    dets = detector.detect(
                            image_[:, :, ::-1]
                        )[:, :4]
    for bbox in dets:
        top, right, bottom, left = [int(_) for _ in bbox]
        cv2.rectangle(image_, (top, right), (bottom, left), (0, 0, 255), 2)
        image_cropped = image_[right:left, top:bottom]
        image_transformed = transform(TF.to_pil_image(image_cropped)).unsqueeze(0)
        age = model(image_transformed)
        age = int(age.item())
        label = f'age:{age}'
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(image_, label, (bottom-w, right - h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
class AgeEstimationModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()
        self.fc = nn.Linear(in_features=512, out_features=1, bias=True)


    def forward(self, x):
        y = self.fc(self.model(x))
        return y

AgeEstimationModel()
test_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
st.set_page_config(
    page_title="Age Estimation Web App",
    page_icon="🔮",

)

st.title('Age Estimation Web App')

st.markdown("""
Welcome to our Age Estimation Web App! This application utilizes the powerful ResNet model, built on PyTorch,
 to accurately estimate the age of individuals from their photographs.
             Upload an image and let our state-of-the-art deep learning model predict the age in just a few seconds. 
            Experience the power of AI in age estimation!
* **Python libraries:** pytorch, pandas, streamlit, numpy
* **Data source:** [github-repository](https://github.com/shgyg99/Age-Estimation).
""")

st.write('---')


tab1, tab2 = st.tabs(['UPLOAD A PICTURE', 'TAKE A PICTURE'])
with tab1:
    up = st.file_uploader('upload', type=['png', 'jpg'])
    if up is not None:

        a = st.image(up)
        b = st.warning('just a sec')
        out = picture_upload(model, up.read(), test_transform)
        st.balloons()
        time.sleep(1)
        a.image(out)


    
with tab2:
    if 'button' not in st.session_state:
        st.session_state.button = True

    def click_button():
        st.session_state.button = not st.session_state.button

    st.button('ON/OFF', on_click=click_button)

    image = st.camera_input("Take a picture", disabled=st.session_state.button)


    if image is not None:

        a = st.image(image)
        b = st.warning('just a sec')
        out = picture_upload(model, image.read(), test_transform)
        st.balloons()
        time.sleep(1)
        a.image(out)


st.write('------')
st.markdown('[send an email to me](mailto:shgyg99@gmail.com)',  unsafe_allow_html=True)