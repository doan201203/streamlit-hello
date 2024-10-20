from google.cloud import firestore, storage
import requests
from google.cloud.firestore import FieldFilter as fil
import time
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os

class DBHandle:
  def __init__(self, dbname) -> None:
    self.dbName = dbname
    self.db = firestore.Client.from_service_account_info(st.secrets)
    self.bucket = storage.Client.from_service_account_info(st.secrets).get_bucket("demo2-2a1d9.appspot.com")
  
  def insert(self, data: dict):
    try:
      new_ref = self.db.collection(self.dbName).add(data)
      new_reff = (new_ref[0]) 
      print('Time insert:', new_ref[0].ToDatatime())
      return new_ref[1].id
    except Exception as e:
      return False
  
  def update(self, id, data: dict):
    try:
      return self.db.collection(self.dbName).document(id).update(data)
    except Exception as e:
      return False
  
  def get_all(self):
    try:
      return self.db.collection(self.dbName).stream()
    except Exception as e:
      return False
  
  def get_by_id(self, id):
    try:
      return self.db.collection(self.dbName).document(id).get()
    except Exception as e:
      return False
  
  def delete(self, id):
    try:
      return self.db.collection(self.dbName).document(id).delete()
    except Exception as e:
      return False      
  
  def upload_file(self, file: UploadedFile, path: str):
    try:
      blob = self.bucket.blob(os.path.join(path, file.name))
      blob.upload_from_file(file, content_type=file.type)
      return True
    except Exception as e:
      return False