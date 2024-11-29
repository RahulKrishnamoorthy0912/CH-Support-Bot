#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.core import load_index_from_storage,StorageContext
#from llama_index.storage.storage_context import StorageContext


# In[3]:


# Configure OpenAI API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# In[ ]:


# Streamlit App Configuration
st.set_page_config(page_title="CH Support Chatbot", layout="wide")
st.title("Welcome to CH Chatbot")
st.markdown("Ask your support-related queries and get instant answers!")


# In[ ]:


# Load the Prebuilt Index
@st.cache_resource
def load_index():
    try:
        persist_dir = "E:\CricHeroes\1. Data Science\AI Support Chatbot\support_index.json"  # Directory where index is saved
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        #index = VectorStoreIndex.load_from_disk(persist_dir)
        return index
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return None

index = load_index()


# In[ ]:


# Check if the index is loaded
if index:
    query_engine = index.as_query_engine()
    
    # User Query Section
    st.header("Ask a Question:")
    user_query = st.text_input("Type your question here:", placeholder="E.g., How to reset my password?")
    
    if user_query:
        with st.spinner("Fetching response..."):
            try:
                response = query_engine.query(user_query)
                st.subheader("Response:")
                st.write(response.response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("Index is not loaded. Please build and add the index file.")


# In[ ]:

# Footer
st.markdown("---")
st.markdown("Powered by [LlamaIndex](https://gpt-index.readthedocs.io/en/latest/) and OpenAI GPT models.")

