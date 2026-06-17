import re
import unicodedata
from langchain_core.documents import Document


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s.,;:!?''""\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_documents(documents: list[Document]) -> list[Document]:
    normalized = []
    for doc in documents:
        normalized_content = normalize_text(doc.page_content)
        normalized.append(
            Document(page_content=normalized_content, metadata=doc.metadata.copy())
        )
    return normalized
