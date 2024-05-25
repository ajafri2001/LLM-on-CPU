import os
import time
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstore/db_faiss"

# Define the custom prompt template
custom_prompt_template = """### Instructions:
You are a helpful, respectful and honest assistant. If asked a question, answer as helpfully as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

### Context:
{context}

### Question:
{question}

### Answer:
"""


# Function to create a prompt template
def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return prompt


# Function to create a Retrieval QA chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        verbose=True,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# Function to load the LLM
def load_llm():
    # Load environment variables from .env file
    load_dotenv()

    # Get model parameters from environment variables
    model = os.getenv("MODEL")
    model_type = os.getenv("MODEL_TYPE")
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS"))
    temperature = float(os.getenv("TEMPERATURE"))
    max_length = int(os.getenv("MAX_LENGTH"))

    llm = CTransformers(
        model=model,
        model_type=model_type,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_length=max_length,
    )
    return llm


# Function to initialize the QA bot
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


# Function to get the final result based on a query
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


# Chainlit event to start the chat
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Greetings👋! Go ahead and ask away any questions you might have about the source documents!")
    await msg.send()
    await msg.update()
    cl.user_session.set("chain", chain)


# Chainlit event to handle incoming messages
# Chainlit event to handle incoming messages
# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")

#     cb = cl.AsyncLangchainCallbackHandler( stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
#     cb.answer_reached = True
#     res = await chain.acall(message.content, callbacks=[cb])
#     answer = res["result"]

#     # Send the final answer
#     await cl.Message(content=answer).send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    # cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    await cl.Message(content=answer).send()    