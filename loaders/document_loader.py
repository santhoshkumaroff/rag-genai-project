from langchain_community.document_loaders import PyPDFLoader, TextLoader
from normalizer.text_normalizer import normalize_documents


def load_document(path: str):

    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif path.endswith(".txt"):
        loader = TextLoader(path)
    else:
        raise ValueError("Unsupported file format")

    documents = loader.load()
    documents = normalize_documents(documents)

    return documents
