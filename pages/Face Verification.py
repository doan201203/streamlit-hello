import streamlit as st
from google.cloud import firestore

st.set_page_config(page_title="Face Verification", initial_sidebar_state="expanded")
st.title("Face Verification")

db = firestore.Client.from_service_account_json('demo2-2a1d9-firebase-adminsdk-yd2h4-1f3ca9dd12.json')
doc_ref = db.collection('face_dataset').document('Face1')

doc = doc_ref.get()
st.write("The id is: ", doc.id)
st.write("The data is: ", doc.to_dict())
