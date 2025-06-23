import streamlit as st
import cv2
import numpy as np
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import speech_recognition as sr
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import multiprocessing

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

predefined_pdfs = ["Herbal_Plants.pdf","A1.pdf"]

def extract_text_from_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text

def get_pdf_text(pdf_files):
    with multiprocessing.Pool() as pool:
        texts = pool.map(extract_text_from_pdf, pdf_files)
    return "\n".join(texts)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Use the provided context to answer the question. If the context lacks information, suggest related herbs or direct the user to another resource.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=10)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply:", response["output_text"])

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.success(f"You said: {query}")
        return query
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Error connecting to speech recognition service."

def orb_feature_matching(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0  

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    return len(matches)  

def find_best_match(uploaded_image, folder="herbal_images"):
    best_match = None
    best_score = 0

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        comparison_img = cv2.imread(img_path)

        if comparison_img is not None:
            score = orb_feature_matching(uploaded_image, comparison_img)
            if score > best_score:
                best_score = score
                best_match = img_name 

    return best_match, best_score

def get_plant_details(plant_name):
    query = f"Discuss about {plant_name}"
    user_input(query)  

def main():
    arr = []
    st.set_page_config(page_title="HERBSPHERE AI Chatbot", layout="centered")
    st.markdown(
        """
        <style>
            body {
                background-color: #e6ffe6;
            }
            .sidebar .sidebar-content {
                background-color: #c2f0c2;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #006600;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("🌿 HERBSPHERE AI Chatbot 🌱")
    st.markdown(
        """
        Welcome to **HERBSPHERE**, your AI-powered herbal guide 🌱
        
        🔍 **Ask about herbal plants** for remedies and benefits.
        """
    )


    if not os.path.exists("faiss_index"):
        with st.spinner("Processing Herbal Documents..."):
            raw_text = get_pdf_text(predefined_pdfs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)

    user_question = st.text_input("💬 Ask about herbal plants:")
    if user_question:
        user_input(user_question)
        arr.append(user_question)
    if st.button("🎙️ Use Voice Input"):
        voice_query = recognize_speech()
        if voice_query:
            user_input(voice_query)
            arr.append(voice_query)

    uploaded_file = st.file_uploader("📸 Upload a plant image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        best_match, match_score = find_best_match(uploaded_image)

        if best_match:
            plant_name = os.path.splitext(best_match)[0]  
            st.image(os.path.join("herbal_images", best_match))
            get_plant_details(plant_name)
        else:
            st.warning("⚠️ No good match found.")

    st.sidebar.header("🌿 Daily Herbal Insight")
    st.sidebar.write("Did you know? Tulsi is known for boosting immunity and reducing stress!")

if __name__ == "__main__":
    main()
