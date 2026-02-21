import os 
import operator
from dotenv import load_dotenv
import json
import spacy
from langchain_groq import ChatGroq
from typing import TypedDict , Annotated
from pydantic import BaseModel
from langgraph.graph import StateGraph, START ,END 
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Chat history
chat_history = []

# Prompt 
# ------- Standalone Message ---------
standalone_message = (
    "Given the chat history and the latest user question, "
    "rewrite the question so that it is fully standalone, clear, "
    "and searchable. "
    "Do not reference the chat history. "
    "Do not add extra explanation. "
    "Just return the rewritten question."
)

# ------- Document evaluation prompt-----------
evaluation_prompt = ChatPromptTemplate.from_template("""
You are a document relevance evaluator for a Retrieval-Augmented Generation (RAG) system.

Your task:
Given a user question and a retrieved document chunk, evaluate how relevant the document 
is for answering the question.

Instructions:
- Assign a relevance score between 0 and 1.
- 0 = completely irrelevant
- 0.25 = weakly related but not useful
- 0.5 = partially relevant but incomplete
- 0.75 = mostly relevant and useful
- 1 = highly relevant and directly answers the question

Scoring guidelines:
- Consider semantic meaning, not just keyword overlap.
- Penalize vague connections.
- If the document contains direct factual support, score high.
- If it only mentions related terms without answering the question, score low.
- If the document contradicts the question context, score 0.

Respond ONLY in this JSON format:

{{
  "score": <float between 0 and 1>,
  "reason": "<short explanation of why you assigned this score>"
}}

User Question:
{question}

Retrieved Document:
{chunks}
""")
# Prompt for refine the document
refine_prompt = ChatPromptTemplate.from_template("""
You are a document filtering assistant in a Retrieval-Augmented Generation (RAG) system.

Your task:
Given a user question and ONE sentence from a document,
decide whether this sentence should be kept for answering the question.

Rules:
- Keep the sentence ONLY if it directly helps answer the question.
- If the sentence is irrelevant, vague, or unrelated, drop it.
- Be strict in filtering.
Respond ONLY in this JSON format:


{{
  "keep": true or false,
  "reason": "..."
}}

User Question:
{question}

Sentence:
{document}
""")

# Prompt for rewrite the query for web search
query_prompt = ChatPromptTemplate.from_template("""You are a query rewriting assistant for a 
                                                Retrieval-Augmented Generation (RAG) system.

Your task:
Rewrite the user's question into an optimized web search query.

Goals:
- Make the query clear and specific.
- Add important keywords if missing.
- Remove conversational or unnecessary words.
- Preserve the original intent.
- Do NOT change the meaning.
- Do NOT answer the question.

Rules:
- Return a concise search query.
- Avoid full sentences when possible.
- Prefer keyword-style phrasing.
- Include proper nouns if relevant.
- If the question asks for recent information, include terms like "latest", "2024", or "current" when appropriate.
- Do not include explanations.

Return ONLY the rewritten query as plain text.

User Question:
{question}
""")

# ------Answer generate Prompt ----------
answer_prompt = """
You are a Corrective Retrieval-Augmented Generation (CRAG) assistant.

Your task:
Answer the question using ONLY the verified context provided below.

Strict Rules:
1. Use only information explicitly present in the context.
2. Do NOT use prior knowledge.
3. Do NOT hallucinate or assume missing details.
4. If the context contains partial information, answer only what is supported.
5. If the context does not contain enough relevant information, respond exactly with:
   "Insufficient verified information available."

Answer Guidelines:
- Be clear and concise.
- Use short paragraphs or bullet points if helpful.
- Do not mention the word "context" in your answer.
- Do not explain your reasoning process.

Verified Context:
{context}

Question:
{question}

Answer:
"""

# Connect to your vector database
COLLECTION_NAME = "legal_acts"
persistant_directory = "D:\\Project\\rag_embeddings"
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_database = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=persistant_directory,
    embedding_function=embedding_model
)

# Threshold values
upper_thres = 0.8
lower_thres = 0.6

# llm
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY,   
)

# Nlp model
nlp = spacy.load("en_core_web_sm")

# Pydantic model
class Doc_evaluator_schema(BaseModel):
    score: float
    reason: str

# State Graph
class State(TypedDict):
    query: str
    chat_history: Annotated[list[BaseMessage],operator.add]
    search_query: str
    docs: list[Document]
    good_docs: list[Document]
    verdict: str
    reason: str
    strips: list[str]
    kept_strips: list[str]
    refined_docs: str
    web_query: str
    web_documents: list[Document]
    answer: str


# Node for standalne the message from the chat history 
def msg_standalne(state: State) -> str:
    chat_history = state.get("chat_history", [])
    user_question = state["query"]
    if chat_history:
        messages = (
            [SystemMessage(content=standalone_message)]
            + chat_history
            + [HumanMessage(content=f"new_question: {user_question}")]
        )
        result = model.invoke(messages)
        search_query = result.content.strip()

    else:
        search_query = user_question

    return {"search_query":search_query}

# Node for retrieve the answer from the database 
def retriever_node(state: State) ->list[Document]:
    print("Searching for:", state["search_query"])
    
    query = state["search_query"]
    
    docs = vector_database.similarity_search(query, k=5)

    return {
        "docs": docs
    }

# Node for evaluate the docs that retrieves from the database
def doc_evaluator(state: State) -> State:
    question = state["search_query"]
    scores: list[float] = []
    good: list[str] = []


    docs_eval_prompt = evaluation_prompt|model.with_structured_output(Doc_evaluator_schema)

    for d in state["docs"]:
        outcome = docs_eval_prompt.invoke({"question":question,"chunks":d.page_content})
        scores.append(outcome.score)

        if outcome.score > lower_thres :
            good.append(d)
    
    # CORRECT: at least one doc > upper_thres        
    if any(s > upper_thres for s in scores):
        return{
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {upper_thres}."
        }
    # INCORRECT: all docs < lower_thres
    elif len(scores)>0 and all(s < lower_thres for s in scores):
        return{
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunk scored < {lower_thres}."
        }
    # AMBIGUOUS: otherwise
    else:
        return{
            "good_docs": good,
            "verdict": "AMBIGUOUS",
            "reason":  f"No chunk scored > {upper_thres}, but not all were < {lower_thres}."
        }
    
# Router Node
def router_node(state: State):
    if state["verdict"] == "CORRECT":
        return "refine_document"
    else:
        return "rewrite_query_node"    

# Node for refine the context
def decompose_to_sentences(text: str) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

# Node for refine the document
def refine_document(state: State) -> State:
    question = state["search_query"]

    prompt = refine_prompt | model

    if state.get("verdict") == "CORRECT":
        docs_to_use = state["good_docs"]
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state["web_documents"]
    else:
        docs_to_use = state["good_docs"] + state["web_documents"]

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()

    strips = decompose_to_sentences(context)

    kept: list[str] = []

    for s in strips:
        response = prompt.invoke({
            "question": question,
            "document": s
        })

        try:
            parsed = json.loads(response.content.strip())
            if parsed.get("keep") is True:
                kept.append(s)
        except Exception:
            continue  

    refined_context = "\n".join(kept).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_docs": refined_context,
    }

# Node for rewrite the query for web search
def rewrite_query_node(state: State) -> str:
    rewrite_chain = query_prompt | model  

    out = rewrite_chain.invoke({
        "question": state["search_query"]
    })

    return {
        "web_query": out.content.strip()
    }


# Node for web search 
tavily = TavilySearch(
    max_results = 5,
    tavily_api_key = TAVILY_API_KEY
)

def web_search_node(state: State) -> dict:
    q = state.get("web_query") or state["search_query"]

    results = tavily.invoke({"query": q})

    web_docs = []

    search_results = results.get("results", [])

    for r in search_results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")

        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"

        web_docs.append(
            Document(
                page_content=text,
                metadata={"url": url, "title": title}
            )
        )

    return {"web_documents": web_docs}

# Node for generate the answer from final context 
def generate_answer(state: State) -> dict:
    search_question = state["search_query"]
    context = state["refined_docs"]

    prompt = answer_prompt.format(
        context=context,
        question=search_question
    )

    response = model.invoke(prompt)
    answer = response.content.strip()

    # Remember the conversation
    chat_history.append(HumanMessage(content=search_question))
    chat_history.append(AIMessage(content=answer))

   
    return {"answer":answer}


graph = StateGraph(State)

graph.add_node("standalone", msg_standalne)
graph.add_node("retriever", retriever_node)
graph.add_node("document_evaluator",doc_evaluator)
graph.add_node("refine_document",refine_document)
graph.add_node("rewrite_query_node",rewrite_query_node)
graph.add_node("web_search_node",web_search_node)
graph.add_node("generator", generate_answer)

graph.add_edge(START, "standalone")
graph.add_edge("standalone", "retriever")
graph.add_edge("retriever", "document_evaluator")
graph.add_conditional_edges(
    "document_evaluator",
    router_node,
    {"refine_document": "refine_document",
     "rewrite_query_node": "rewrite_query_node",
     }
    )
graph.add_edge("rewrite_query_node","web_search_node")
graph.add_edge("web_search_node","refine_document")
graph.add_edge("refine_document","generator")
graph.add_edge("generator", END)

app = graph.compile()

def ask_rag(question: str) -> str:
   initial_state = {
    "query": question,
    "chat_history": chat_history,
    "search_query": "",
    "docs": [],
    "good_docs": [],
    "web_documents": [],
    "refined_docs": "",
    "verdict": "",
    "reason": "",
    "answer": ""
}
   result = app.invoke(initial_state)
   return result["answer"]


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (type 'exit' to stop): ")
        
        if question.lower() == "exit":
            break
        
        answer = ask_rag(question)
        print("\nAnswer:\n", answer)
