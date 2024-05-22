from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstore/db_faiss"

# Define the custom prompt template
custom_prompt_template = """### Instructions:
Use the following context to answer the question. If you don't know the answer, just say so.

### Context:
{context}

### Question:
{question}

### Answer:
"""

# Function to create a prompt template
def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=["context", "question"]
    )
    return prompt

# Function to create a Retrieval QA chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

# Function to load the LLM
def load_llm():
    llm = CTransformers(
        model="models/llama-2-7b.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.5,
    )
    return llm

# Function to initialize the QA bot
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Function to get the final result based on a query
def final_result(query):
    qa = qa_bot()
    response = qa({"query": query})
    answer = response["result"]

    # Remove prefix tokens from the answer
    prefix_tokens = ["FINAL", "ANSWER"]
    for token in prefix_tokens:
        answer = answer.replace(token, "").strip()

    sources = response.get("source_documents", [])
    if sources:
        sources_str = "\n".join([f"- {doc.metadata['source']}, page {doc.metadata['page']}" for doc in sources])
        answer += f"\nSources:\n{sources_str}"
    else:
        answer += "\nNo sources found"

    return answer

# Chainlit event to start the chat
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

# Chainlit event to handle incoming messages
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
      answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res.get("source_documents", [])
    
    if sources:
        sources_str = "\n".join([f"- {doc.metadata['source']}, page {doc.metadata['page']}" for doc in sources])
        answer += f"\nSources:\n{sources_str}"
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
