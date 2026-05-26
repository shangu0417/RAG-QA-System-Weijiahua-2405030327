import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="RAG问答系统", layout="wide")
st.title("📚 RAG智能问答系统")

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_chunk_count():
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)._collection.count()
    except:
        return 0

def build_kb():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        return False
    docs = []
    for f in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, f)
        if f.endswith('.pdf'):
            docs.extend(PyPDFLoader(path).load())
        elif f.endswith('.docx'):
            docs.extend(UnstructuredWordDocumentLoader(path).load())
    if not docs:
        return False
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR).persist()
    return True

with st.sidebar:
    st.header("知识库管理")
    st.metric("文本块数量", get_chunk_count())
    uploaded = st.file_uploader("上传文档", type=["pdf", "docx"])
    if uploaded:
        os.makedirs(DOCS_DIR, exist_ok=True)
        with open(os.path.join(DOCS_DIR, uploaded.name), "wb") as f:
            f.write(uploaded.getbuffer())
        if build_kb():
            st.success("知识库已更新")
            st.rerun()
    if st.button("重建知识库"):
        if build_kb():
            st.success("重建成功")
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if get_chunk_count() == 0:
        response = "请先上传文档"
    else:
        llm = Ollama(model="deepseek-r1:7b")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        prompt_template = PromptTemplate(template="基于文档回答：\n文档：{context}\n问题：{question}\n回答：", input_variables=["context", "question"])
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), chain_type_kwargs={"prompt": prompt_template})
        response = qa.invoke(prompt)["result"]
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})