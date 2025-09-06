
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def unified_chunking(text, source="Unknown", lang="en", chunk_size=1000, chunk_overlap=100):
    """
    Chunk legal/rights documents into semantically meaningful pieces with CCH.
    Each chunk gets metadata: source, header, body.
    """
    chunks = []

    if text is None:  
        return chunks

    if lang == "en":
        article_pattern = r"(Article\s+\d+[A-Z]?(?::.*?)?)"
    else:  # Bangla
        article_pattern = r"(অনুচ্ছেদ\s+\d+[A-Z]?(?::.*?)?)"

    if re.search(article_pattern, text):
        parts = re.split(article_pattern, text)
        current_header = "General"
        for i in range(len(parts)):
            if parts[i] is not None and re.match(article_pattern, parts[i]):
                current_header = parts[i].strip()
            else:
                body = parts[i].strip() if parts[i] is not None else "" 
                if body:
                    chunks.append({
                        "text": f"Header: {current_header}\nBody: {body}",
                        "metadata": {"source": source, "header": current_header}
                    })
        return chunks

    pattern = r"(Chapter\s+\d+.*|Section\s+\d+(\.\d+)*.*|Part\s+[A-Z]+.*)"
    if re.search(pattern, text):
        parts = re.split(pattern, text)
        current_header = "General"
        for i in range(len(parts)):
            if parts[i] is not None and re.match(pattern, parts[i]): # Add check for None
                current_header = parts[i].strip()
            else:
                body = parts[i].strip() if parts[i] is not None else "" # Add check for None
                if body:
                    chunks.append({
                        "text": f"Header: {current_header}\nBody: {body}",
                        "metadata": {"source": source, "header": current_header}
                    })
        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ".", " "]
    )
    text_chunks = splitter.split_text(text)
    for c in text_chunks:
        chunks.append({
            "text": f"Header: General Section\nBody: {c}",
            "metadata": {"source": source, "header": "General Section"}
        })
    return chunks

from load_pdf import load_pdf_directory
def chunking():
  constitution_data, Right_and_law_data = load_pdf_directory()
  c_chunk = []
  for doc in constitution_data:
    c_chunk.extend(unified_chunking(doc.page_content, source="constitution"))

  r_chunk = []
  for doc in Right_and_law_data:
    r_chunk.extend(unified_chunking(doc.page_content, source="Right and law"))
  print(f"Total constitution chunks: {len(c_chunk)}")
  return c_chunk,r_chunk

if __name__ == "__main__":
  c_chunk,r_chunk = chunking()
  print(len(c_chunk))
  print(len(r_chunk))