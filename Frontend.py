import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
st.set_page_config(page_title='Face Mask Detection',page_icon='https://5.imimg.com/data5/PI/FD/NK/SELLER-5866466/images-500x500.jpg')
st.title('Welcome to FaceMask Detection 😷')
s=st.sidebar.selectbox('Home',('Home','IP camera','File'))
st.sidebar.image('https://cdn.hackernoon.com/images/oO6rUouOWRYlzw88QM9pb0KyMIJ3-bxfy3m27.png')
if(s=='Home'):
    st.image('https://miro.medium.com/v2/resize:fit:2000/0*gvwpXO8cpZOY5BeS')
elif(s=='IP camera'):
    url=st.text_input('enter the url')
    window=st.empty()
    btn=st.button('start')
    if btn:
        mask_model=load_model('model.h5',compile=False)
        facedet=cv2.CascadeClassifier('face.xml')
        v=cv2.VideoCapture(url)
        btn2=st.button('Stop')
        if btn2:
            v.release()
            st.experimental_rerun()
        while True:
            flag=v.read()
            f=v.read()
            if flag:
                fm=facedet.detectMultiScale(f)
                for(x,y,l,w)in fm:
                    face_img=f[y:y+w,x:x+l]
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
                    face_img=(face_img/127.5)-1
                    pred=mask_model.predict(face_img)[0][0]
                    if(pred>0.9):
                        cv2.rectangle(f,(x,y),(l+x,w+y),(0,0,0),3)
                    else:
                        cv2.rectangle(f,(x,y),(l+x,w+y),(255,255,255),3)
                window.image(f,channels='BGR')

elif(s=='File'):
    u=st.text_input('enter the URL or file name')
    window=st.empty()
    btn=st.button('start')
    if btn:
        mask_model=load_model('model.h5',compile=False)
        facedet=cv2.CascadeClassifier('face.xml')
        v=cv2.VideoCapture(int(u))
        btn2=st.button('Stop')
        if btn2:
            v.release()
            st.experimental_rerun()
        while True:
            flag,f=v.read()
            if flag:
                fm=facedet.detectMultiScale(f)
                for(x,y,l,w)in fm:
                    face_img=f[y:y+w,x:x+l]
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
                    face_img=(face_img/127.5)-1
                    pred=mask_model.predict(face_img)[0][0]
                    if(pred>0.9):
                        cv2.rectangle(f,(x,y),(l+x,w+y),(0,0,0),3)
                    else:
                        cv2.rectangle(f,(x,y),(l+x,w+y),(255,255,255),3)
                window.image(f,channels='BGR')

        
