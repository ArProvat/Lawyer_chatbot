
from langchain_community.document_loaders import PyPDFDirectoryLoader
def load_pdf_directory():
  constitution_loader = PyPDFDirectoryLoader(r"Data\constitution")
  Right_and_law_loader = PyPDFDirectoryLoader(r"Data\law and right")
  constitution_data = constitution_loader.load()
  Right_and_law_data = Right_and_law_loader.load()
  return constitution_data, Right_and_law_data

if __name__ == "__main__":
    constitution_data, Right_and_law_data = load_pdf_directory()
    print(f"Total constitution documents: {len(constitution_data)}")
    print(f"Total Right and law documents: {len(Right_and_law_data)}")