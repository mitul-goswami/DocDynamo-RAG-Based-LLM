import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import subprocess

# Ensure correct version of protobuf is installed
subprocess.run(["pip", "install", "protobuf==3.20.3"])

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

    model = ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY)  # Pass the API key here
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_ip(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})

    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)  # Enable dangerous deserialization
    docs = new_db.similarity_search(user_question, k=3)

    chain = conv_chain()

    response = chain(
        {'input_documents': docs, "question": user_question}
        , return_only_outputs=True
    )
    
    print(response)

    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF - OpenRAG💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_ip(user_question)

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
    
    st.markdown("### Exciting Update for OpenRAG Users!")
    st.markdown("""
    We are thrilled to introduce the latest feature added to DocDynamo: **Langvault** 📚✨.
    
    Now you can convert your PDFs between international and regional languages, and vice versa, with just two clicks! Experience seamless language conversion and enhance your document accessibility today.
    """)
    
    st.link_button("Experience LangVault", "https://pdf-translator--openrag.streamlit.app/")
        

if __name__ == "__main__":
    main()
