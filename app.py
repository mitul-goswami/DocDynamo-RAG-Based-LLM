import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import subprocess
import requests
from bs4 import BeautifulSoup
import json
from youtube_transcript_api import YouTubeTranscriptApi

# Ensure correct version of protobuf is installed
subprocess.run(["pip", "install", "protobuf==3.20.3"])

import spacy
from spacy.cli import download

def ensure_spacy_model(model_name="en_core_web_sm"):
    try:
        spacy.load(model_name)
    except OSError:
        download(model_name)
        # Load the model again after downloading
        spacy.load(model_name)

# Ensure the spaCy model is installed
ensure_spacy_model()
nlp = spacy.load("en_core_web_sm")

GROQ_API_KEY = "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"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def conv_chain():
    prompt_template = """ 
Answer the question as detailed as possible from the provided context keeping the tone professional and 
acting like an expert and if you don't know the answer, just say "Answer is not there within the context", don't provide the wrong answer \n\n
Context: \n {context}?\n
Question: \n{question}\n

Answer:

    """

    model = ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_ip(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})

    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=3)

    chain = conv_chain()

    response = chain(
        {'input_documents': docs, "question": user_question}
        , return_only_outputs=True
    )

    return response["output_text"], docs

def get_video_ids(user_query):
    headers = {"User-Agent": "Guest"}
    video_res = requests.get(f'https://www.youtube.com/results?search_query={"+".join(user_query.split(" "))}', headers=headers)
    soup = BeautifulSoup(video_res.text, 'html.parser')
    arr_video = soup.find_all('script')

    arr_main = []
    for i in arr_video:
        if 'var ytInitialData' in str(i.get_text()):
            arr_main.append(i)
            break
    main_script = arr_main[0].get_text()[arr_main[0].get_text().find('{'):arr_main[0].get_text().rfind('}')+1]
    data = json.loads(main_script)
    video_data = data.get('contents').get('twoColumnSearchResultsRenderer').get('primaryContents').get('sectionListRenderer').get('contents')[0].get('itemSectionRenderer').get('contents')
    video_json = [i for i in video_data if 'videoRenderer' in str(i)]

    video_ids = [i.get('videoRenderer').get('videoId') for i in video_json if i.get('videoRenderer')]
    return video_ids

def get_transcript(video_ids):
    yt_data = []
    for i in video_ids:
        txt = ""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(i, languages=['en'])
            for j in transcript:
                txt += j['text'] + " "
            yt_data.append({"video_id": i, "transcript": txt})
        except:
            continue
    return yt_data

def compare_transcript(arr):
    clean_arr = []
    for i in arr:
        transcript = i['transcript']
        doc = nlp(transcript)
        filtered_words = [token.text for token in doc if not token.is_stop]
        clean_transcript = ' '.join(filtered_words)
        clean_arr.append({"video_id": i['video_id'], "transcript": clean_transcript})
    return clean_arr

def video_recommendation(user_question):
    video_ids = get_video_ids(user_question)
    transcripts = get_transcript(video_ids)
    clean_transcripts = compare_transcript(transcripts)
    
    # Sort videos based on the length of clean transcript as a proxy for relevance
    sorted_videos = sorted(clean_transcripts, key=lambda x: len(x['transcript']), reverse=True)
    
    return sorted_videos[:4]  # Return top 4 videos

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF - OpenRAG💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        response, docs = user_ip(user_question)
        st.write("Reply: ", response)
        
        # Extract text from the top 3 documents to provide context
        context_text = " ".join([doc.page_content for doc in docs])

        # Fetch video recommendations based on the user question and PDF context
        video_query = f"{response} {context_text}"
        recommendations = video_recommendation(video_query)
        
        st.subheader("YouTube Video Recommendations:")
        if recommendations:
            for i in range(0, len(recommendations), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(recommendations):
                        st.write(f"Video ID: {recommendations[i]['video_id']}")
                        st.video(f"https://www.youtube.com/watch?v={recommendations[i]['video_id']}")
                
                with col2:
                    if i + 1 < len(recommendations):
                        st.write(f"Video ID: {recommendations[i+1]['video_id']}")
                        st.video(f"https://www.youtube.com/watch?v={recommendations[i+1]['video_id']}")
        else:
            st.write("No YouTube videos found for this query.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                st.write("")
                st.write("### LangVault 📑")
            st.write("**Major update for OpenRAG users**")

    st.markdown("---")
    
    st.markdown("### Exciting Update for DocDynamo Users!")
    st.markdown("""
    We are thrilled to introduce the latest feature added to DocDynamo: **Langvault** 📚✨.
    
    Now you can convert your PDFs between international and regional languages, and vice versa, with just two clicks! Experience seamless language conversion and enhance your document accessibility today.
    """)
    
    st.link_button("Experience LangVault", "https://pdf-translator--openrag.streamlit.app/")

if __name__ == "__main__":
    main()
