from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from prompts.rag_prompt import SYSTEM_PROMPT, RAG_PROMPT
from models.schemas import RAGQueryOutput


def build_rag_chain(retriever, llm):

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def extract_sources(docs):
        return [doc.metadata.get("source", "unknown") for doc in docs]

    full_prompt = SYSTEM_PROMPT + "\n" + RAG_PROMPT
    prompt = PromptTemplate.from_template(full_prompt)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
