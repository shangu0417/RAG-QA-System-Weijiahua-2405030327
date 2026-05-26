from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="基于以下文档回答问题。如果文档中没有，就说无法回答。\n文档：{context}\n问题：{question}\n回答：",
    input_variables=["context", "question"]
)

def ask(question):
    llm = Ollama(model="deepseek-r1:7b")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs={"prompt": prompt})
    return qa.invoke(question)["result"]

if __name__ == "__main__":
    while True:
        q = input("问题：")
        if q == "exit":
            break
        print("回答：", ask(q))