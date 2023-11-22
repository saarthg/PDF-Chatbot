from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import openai
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from pdf import pdf_qa
from youtube import youtube_qa

# Load environment variables
load_dotenv(".env")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY


def main():
    st.title("Chat with your Documents/Videos 💬")

    def click_button():
        st.session_state.clicked = True
    def click_button2():
        st.session_state.clicked = False
    

    
    pdf_button = st.button("Upload a PDF", on_click = click_button)
    youtube_button = st.button("YouTube Link", on_click = click_button2)

    if 'clicked' in st.session_state and st.session_state.clicked:
        pdf_qa()
    elif 'clicked' in st.session_state and not st.session_state.clicked:
        youtube_qa()
        

if __name__ == "__main__":
    main()
