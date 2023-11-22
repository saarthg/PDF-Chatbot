from dotenv import load_dotenv
import os
import openai
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


load_dotenv(".env")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase

def pdf_qa():
    pdf = st.file_uploader('Upload your PDF Document(s)', type='pdf')

    if pdf is not None:
        text = ""
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        for page in pdf_reader.pages:
            text += page.extract_text()
        # Create the knowledge base object
        knowledgeBase = process_text(text)
        
        query = st.text_input('Ask a question to the PDF')
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()
        
        if query and query=="summarize" or query=="Summarize":
            prompt = PromptTemplate(
                input_variables=["text"],
                template="Write a summary for the following text: {text}",
            )
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
            chain = LLMChain(llm=llm, prompt=prompt)
            st.write(chain.run(text))
        elif query:
            docs = knowledgeBase.similarity_search(query)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type='stuff')
            
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
                
                st.write(response)