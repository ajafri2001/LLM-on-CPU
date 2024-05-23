import time
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstore/db_faiss"

# Define the custom prompt template
custom_prompt_template = """### Instructions:
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

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
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# Function to load the LLM
def load_llm():
    llm = CTransformers(
        model="models/llama-2-7b-chat.Q5_K_M.gguf",
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.5,
        max_length=1024,  # Increase this value to allow for a larger context length
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
    qa = qa_bot()
    response = qa({"query": query})
    answer = response["result"]

    # Remove prefix tokens from the answer
    prefix_tokens = ["FINAL", "ANSWER"]
    for token in prefix_tokens:
        answer = answer.replace(token, "").strip()

    sources = response.get("source_documents", [])

    # Check if the answer is meaningful
    if (
        "I don't know" not in answer
        and "I'm not sure" not in answer
        and "I'm not able" not in answer
    ):
        if sources:
            sources_str = "\n".join(
                [
                    f"- {doc.metadata['source']}, page {doc.metadata['page']}"
                    for doc in sources
                ]
            )
            answer += f"\nSources:\n{sources_str}"
    else:
        answer += "\nNo relevant information found."

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
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    start_time = time.time()  # Record the start time
    res = await chain.acall(message.content, callbacks=[cb])
    end_time = time.time()  # Record the end time
    time_taken = end_time - start_time  # Calculate the time taken

    answer = res["result"]
    sources = res.get("source_documents", [])

    if sources:
        sources_str = "\n".join(
            [
                f"- {doc.metadata['source']}, page {doc.metadata['page']}"
                for doc in sources
            ]
        )
        answer += f"\nSources:\n{sources_str}"
    else:
        answer += "\nNo sources found"

    answer += f"\n\nTime taken: {time_taken:.2f} seconds"  # Append the time taken to the answer

    await cl.Message(content=answer).send()
