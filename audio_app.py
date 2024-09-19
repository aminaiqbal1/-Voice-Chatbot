from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import streamlit as st
import re
import os

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

STATIC_IMAGE_URL = "https://media.istockphoto.com/id/1314799930/vector/voice-assistant-concept-vector-sound-wave-voice-and-sound-recognition-equalizer-wave-flow.jpg?s=612x612&w=0&k=20&c=POoo1NOsA5mYe-E-hA9xxLqGHKzlmSxCdwkDMGeJMlw="

with st.sidebar:
    st.image(STATIC_IMAGE_URL, use_column_width=True)
    pdf_file = st.file_uploader("Upload a pdf file", accept_multiple_files=False)

    # process the uploaded file
    if pdf_file and st.button("Process", type="primary", use_container_width=True):
        with st.spinner("Creating Vectorstore..."):
            temp_file = "./temp.pdf"
            #store the file temporarily
            with open(temp_file, "wb") as file:
                file.write(pdf_file.getvalue())
                file_name = pdf_file.name

            pdf_pages = PyPDFLoader(temp_file).load()  # load all pages
            st.write(f"Total PDF pages: {len(pdf_pages)}")
            # split the pages into smaller chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            text_chunks = splitter.split_documents(pdf_pages)
            st.write(f"Total Chunks: {len(text_chunks)}")
            # create vectorstore
            st.session_state.vectorstore = FAISS.from_documents(text_chunks, CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key="zRSY3f59ElzIFItOFGZb4DvGBvIuiqlWmDJ758Os"))
        st.success("Vectorstore Created! Now you can ask questions.")


API_KEY = "AIzaSyDGmiz57W57FfGlpX5oN_F2qidHDG9_86Q" 

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Get the response from LLM
def generate_answer(question):
    system_prompt = """
    You are an assistant for question-answering tasks in Urdu Language.
    Use the following pieces of retrieved context to answer the user question in accurate Urdu language.
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: {question} 
    -----
    Context: {context} 
    -----
    Answer:""" 

    prompt = ChatPromptTemplate.from_template(system_prompt)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
    retriever = (st.session_state.vectorstore).as_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain.invoke(question)


# Streamlit app layout
st.title("Voice Assistant ChatBot")


# Custom CSS for colorful chat bubbles with blue and purple
st.markdown(
    """
    <style>
    .user-bubble {
        background: linear-gradient(135deg, #1e3c72, #2a5298); /* Blue gradient */
        color: white;
        padding: 10px;
        border-radius: 15px;
        text-align: right;
        margin-left: 25%;
        margin-bottom: 10px;
        font-family: Arial, sans-serif;
    }
    .ai-bubble {
        background: linear-gradient(135deg, #9b59b6, #8e44ad); /* Purple gradient */
        color: white;
        padding: 10px;
        border-radius: 15px;
        text-align: left;
        margin-right: 25%;
        margin-bottom: 10px;
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Voice recording
st.subheader("Record Your Voice:")
question = speech_to_text(language="ur", use_container_width=True, just_once=True, key="STT")


if st.session_state.vectorstore:
    if question:
        st.subheader("Text Generating")
        with st.spinner("Converting to Speech..."):
            try:
                # Get the response from the model
                response = generate_answer(question)
                
                # Clean the response to remove unwanted characters like '**'
                full_response = "".join(res or "" for res in response)
                cleaned_response = re.sub(r"\*\|__", "", full_response)

                # Display the conversation in colorful chat bubbles
                st.markdown(f'<div class="user-bubble">{question}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ai-bubble">{cleaned_response}</div>', unsafe_allow_html=True)

                # Convert cleaned text to speech
                tts = gTTS(text=cleaned_response, lang='ur')
                output_file = "output.mp3"
                tts.save(output_file)
                st.audio(output_file, autoplay=True)

                # Add download option for the generated voice output
                with open(output_file, "rb") as audio_file:
                    st.download_button(label="Download", data=audio_file, file_name="generated_speech.mp3")

            except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    st.info("Upload a pdf file and click 'Process' button.")