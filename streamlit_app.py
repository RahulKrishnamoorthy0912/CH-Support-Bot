#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, get_response_synthesizer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import os

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]


# Initialize LLM and Embedding Models
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-3.5-turbo")  # Replace with the appropriate model

# Streamlit Web App
st.set_page_config(page_title="CH Support Chatbot", layout="wide")
st.title("Welcome to CH Chatbot")
st.markdown("Ask your support-related queries and get instant answers!")

# Sidebar for options
option = st.sidebar.selectbox("Choose an option", ["Upload Support Data", "Chat with Bot"])

if option == "Upload Support Data":
    st.header("Step 1: Upload Your Support Data")
    uploaded_file = st.file_uploader("Upload a support file (PDF format)", type=["pdf"])

    if uploaded_file:
        # Save file temporarily
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and display the document
        documents = SimpleDirectoryReader(input_files=[temp_file]).load_data()
        st.success(f"Successfully ingested {len(documents)} document(s)!")
        
        if documents:
            st.subheader("Preview of First Document:")
            st.text_area("Content", documents[0].text[:500], height=300)
        
        # Build index and store in session state
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, llm=llm)
        st.session_state["index"] = index
        st.success("Index built successfully! You can now query the chatbot.")

elif option == "Chat with Bot":
    st.header("Step 2: Chat with the Support Chatbot")

    # Ensure index is available
    if "index" in st.session_state:
        # Retrieve query engine
        query_engine = st.session_state["index"].as_query_engine(llm=llm,response_synthesizer=get_response_synthesizer(llm=llm))

        # Input query
        user_query = st.text_input("Enter your question:")

        if user_query:
            # Get response
            response = query_engine.query(user_query).response
            st.subheader("Bot Response:")
            st.write(response)
    else:
        st.warning("Please upload support data and build the index first.")

# Footer
st.markdown("---")
st.markdown("Powered by [LlamaIndex](https://gpt-index.readthedocs.io/en/latest/) and OpenAI GPT models.")

