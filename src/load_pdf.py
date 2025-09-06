
from langchain.document_loaders import PyPDFDirectoryLoader
def load_pdf_directory():
  constitution_loader = PyPDFDirectoryLoader("/content/drive/MyDrive/constitution/constitution")
  constitution_data = constitution_loader.load()
  Right_and_law_loader = PyPDFDirectoryLoader("/content/drive/MyDrive/constitution/law and right")
  Right_and_law_data = Right_and_law_loader.load()
  return constitution_data, Right_and_law_data