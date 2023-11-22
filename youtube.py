from dotenv import load_dotenv
import os
import openai
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
from langchain.chat_models import ChatOpenAI

load_dotenv(".env")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY


def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="text-davinci-003")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs

def youtube_qa():
    url = st.text_input("YouTube URL")
    query = st.text_input("What is your question?")
    if query and url and (query=="Summarize" or query=="summarize"):
        loader = YoutubeLoader.from_youtube_url(url)
        transcript = loader.load()
        prompt = PromptTemplate(
                input_variables=["text"],
                template="Write a summary for the following text: {text}",
            )
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        chain = LLMChain(llm=llm, prompt=prompt)
        st.write(chain.run(transcript))
    elif query and url:
        db = create_db_from_youtube_video_url(url)
        response, docs = get_response_from_query(db, query)
        st.write(response)

