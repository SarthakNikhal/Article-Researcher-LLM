import os
import nltk
import streamlit as st
import pickle
import time
import langchain
# from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader,PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from utils.document_loader import load_documents

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

from dotenv import load_dotenv
load_dotenv()  ##take environment variable from .env

st.title("Finance Article Reader")

st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_urls = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placefolder = st.empty()

# llm = ChatOpenAI(
#     model="gpt-3.5-turbo",  # or "gpt-4"
#     temperature=0.7,
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
#     verbose=False,  # Disable verbose logging that might trigger the issue
#     max_tokens = 500
# )
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_urls:
    # load data
    loader = UnstructuredURLLoader(urls = urls)
    main_placefolder.text("Data Feeding...Begins")
    print("Debug checkpoint: about to load loader")
    data = loader.load()
    print("Debug checkpoint: loaded loader")
    print("Data is: ", data)
  

    main_placefolder.text("Text Splitter...Started.")
    # documents = load_documents(urls)
    # # Debug: Check if documents are loaded
    # if documents:
    #     print(f"Successfully loaded {len(documents)} documents.")
    #     for doc in documents[:3]:  # Print a sample of the content
    #         print(f"Content: {doc.page_content[:500]}")  # Show first 500 characters
    # else:
    #     print("No documents loaded. Check the URLs or loaders.")

    # split data
    main_placefolder.text("Text Splitting...Begins")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",".",","," "],
        chunk_size=1000,
        # chunk_overlap=200,
    )

    docs = text_splitter.split_documents(data) # documents

    main_placefolder.text("Creating Embeddings.....")
    embeddings = OpenAIEmbeddings()

    # check if docs are loaded
    if not docs:
        raise ValueError("Documents are empty. Ensure document loading is successful.")
    if not embeddings:
        raise ValueError("Embeddings are empty. Ensure embeddings are being generated.")

    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding Vector Started Building...")
    time.sleep(2)
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placefolder.text_input("Question: ")
if query:
    file_path = "vector_index.pkl"
    embeddings = OpenAIEmbeddings()
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            # Load the FAISS index
            vectorstore = pickle.load(f)

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                chain_type="refine"
            )
            # langchain.debug = True

            # Run the query through the chain
            response = chain({"query": query}, return_only_outputs=True)

            print(response)

            st.header("Answer")
            st.write(response["result"])

            # Display sources, if available
            sources = response.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)