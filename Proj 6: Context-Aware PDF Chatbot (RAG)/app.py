import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="PDF Insight AI", page_icon="📚")
st.title("Document Intelligence Bot 🤖")

api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file and api_key:
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document... This involves chunking and embedding."):
        loader = PyPDFLoader("temp_doc.pdf")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma.from_documents(chunks, embeddings)

        # Initialize the LLM (Gemini 1.5 Flash)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        # Create the Retrieval Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vector_store.as_retriever()
        )

    st.success("Document processed! Ask anything below.")
    user_query = st.text_input("What would you like to know from this PDF?")
    
    if user_query:
        with st.spinner("Searching document for answers..."):
            response = qa_chain.invoke(user_query)
            st.markdown(f"### AI Answer:\n{response['result']}")

elif not api_key:
    st.info("Please enter your Google API Key in the sidebar to begin.")
