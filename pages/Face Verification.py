import streamlit as st
from google.cloud import firestore, storage
import json
import toml
import pandas as pd
from datetime import timedelta

st.set_page_config(page_title="Face Verification", initial_sidebar_state="expanded")
st.title("Face Verification")

db = firestore.Client.from_service_account_info(st.secrets)
bucket = storage.Client.from_service_account_info(st.secrets).get_bucket('demo2-2a1d9.appspot.com')
# db = firestore.Client.from_service_account_json('demo2-2a1d9-firebase-adminsdk-yd2h4-1f3ca9dd12.json')
# bucket = storage.Client.from_service_account_json('demo2-2a1d9-firebase-adminsdk-yd2h4-1f3ca9dd12.json').get_bucket('demo2-2a1d9.appspot.com')

doc_ref = db.collection('face_dataset').stream()
tb = {
  "msv" : [],
  "name" : [],
  "TheSV" : [],
  "ChanDung" : [],
}

for i in doc_ref:
    tb["msv"].append(i.to_dict()["msv"])
    tb["name"].append(i.to_dict()["name"])
    path1 = i.to_dict()["TheSV"].replace("gs://demo2-2a1d9.appspot.com/","")
    path2 = i.to_dict()["ChanDung"].replace("gs://demo2-2a1d9.appspot.com/","")
    
    public_url = bucket.blob(path1).generate_signed_url(expiration=timedelta(seconds=300), method='GET')
    tb["TheSV"].append(f"<img src='{public_url}' width='100'>")
    
    public_url = bucket.blob(path2).generate_signed_url(expiration=timedelta(seconds=300), method='GET')
    tb["ChanDung"].append(f"<img src='{public_url}' width='100'>")
    
    print(path1, path2)
# sto_ref = 
st.header("1. CSDL")
st.write(pd.DataFrame(tb).to_html(escape=False, index=False), unsafe_allow_html=True)
