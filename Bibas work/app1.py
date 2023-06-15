import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image



pickle_in = open("regression_model.pkl","rb")
classifier=pickle.load(pickle_in)


def welcome():
    return "Welcome All"


def predict_(T,TM,Tm,SLP,H,VV,V,VM):
    

    input_df = pd.DataFrame(
    {
               
        "T" : [T],
        "TM" : [TM],
        "Tm" : [Tm],
        "SLP" : [SLP],
        "H" : [H],
        "VV" : [VV],
        "V" : [V],
        "VM" : [VM]
    }
    )
    
    prediction=classifier.predict(input_df)
    print(prediction)
    return prediction



def main():
    st.title("AQI")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    T = st.text_input("T","Type Here")
    TM = st.text_input("TM","Type Here")
    Tm = st.text_input("Tm","Type Here")
    SLP = st.text_input("SLP","Type Here")
    H = st.text_input("H","Type Here")
    VV = st.text_input("VV","Type Here")
    V = st.text_input("V","Type Here")
    VM = st.text_input("VM","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_(T,TM,Tm,SLP,H,VV,V,VM)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Go")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    