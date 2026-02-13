import os 
import operator
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict , Annotated
from langgraph.graph import StateGraph, START ,END 
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Chat history
chat_history = []


# Connect to your vector database
COLLECTION_NAME = "legal_acts"
persistant_directory = "D:\\Project\\rag_embeddings"
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_database = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=persistant_directory,
    embedding_function=embedding_model
)

# llm
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY,   
)


# State Graph
class State(TypedDict):
    query : str
    chat_history : Annotated[list[BaseMessage],operator.add]
    answer : str
    search_question: str
    context : list[str]


# Node

# ------- Standalone Message ---------
standalone_message = (
    "Given the chat history and the latest user question, "
    "rewrite the question so that it is fully standalone, clear, "
    "and searchable. "
    "Do not reference the chat history. "
    "Do not add extra explanation. "
    "Just return the rewritten question."
)



def msg_standalne(state: State) -> dict:
    chat_history = state.get("chat_history", [])
    user_question = state["query"]
    if chat_history:
        messages = (
            [SystemMessage(content=standalone_message)]
            + chat_history
            + [HumanMessage(content=f"new_question: {user_question}")]
        )
        result = model.invoke(messages)
        search_question = result.content.strip()

    else:
        search_question = user_question

    print(f"searching for :{search_question}")
    return {"search_question":search_question}


def retriever_node(state: State) -> dict:
    search_question = state["search_question"]

    # retrieve top 5 with similarity scores
    results = vector_database.similarity_search_with_score(
        search_question, k=5
    )

    THRESHOLD = 0.35  

    # keep only truly relevant chunks
    filtered = [doc for doc, score in results if score < THRESHOLD]

    context = [doc.page_content for doc in filtered]

    print(f"Found {len(context)} relevant documents")

    return {
        "context": context,
        "search_question": search_question
    }


# -------LLM Prompt
answer_prompt = (
    "You are an AI assistant that answers user questions using the provided context.\n\n"
    "Rules:\n"
    "1. Dont use external resources give information from given context only"
    "2. If the context does not contain the answer, say: "
    "'The provided documents do not contain enough information to answer this question.'\n"
    "3. Do NOT make up facts.\n"
    "4. Write in simple, clear, and easy-to-understand language.\n"
    "5. Structure the answer with short paragraphs or bullet points.\n"
    "6. If helpful, give a small example.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)


def generate_answer(state: State) -> dict:
    search_question = state["search_question"]
    context = state["context"]

     # HARD STOP if no docs
    if not context:
        return {
            "answer": " No relevant information found in the provided legal documents."
        }

    prompt = answer_prompt.format(
        context=context,
        question=search_question
    )

    message = [SystemMessage(content=prompt)]
    response = model.invoke(message)
    answer = response.content.strip()

    # Remember the conversation
    chat_history.append(HumanMessage(content=state["query"]))
    chat_history.append(AIMessage(content=answer))

   
    return {"answer":answer}


graph = StateGraph(State)

graph.add_node("standalone", msg_standalne)
graph.add_node("retriever", retriever_node)
graph.add_node("generator", generate_answer)

graph.add_edge(START, "standalone")
graph.add_edge("standalone", "retriever")
graph.add_edge("retriever", "generator")
graph.add_edge("generator", END)

app = graph.compile()

def ask_rag(question: str) -> str:
    initial_state = {
        "query": question,
        "chat_history": chat_history,
        "answer": "",
        "search_question": "",
        "context": []
    }
    result = app.invoke(initial_state)
    return result["answer"]
