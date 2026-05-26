import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"

def build():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print("请将文档放入 docs 文件夹")
        return
    
    docs = []
    for f in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, f)
        if f.endswith('.pdf'):
            docs.extend(PyPDFLoader(path).load())
        elif f.endswith('.docx'):
            docs.extend(UnstructuredWordDocumentLoader(path).load())
    
    if not docs:
        print("未找到文档")
        return
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR).persist()
    print(f"完成！共{len(chunks)}个文本块")

if __name__ == "__main__":
    build()