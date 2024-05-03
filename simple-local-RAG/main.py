import os
import requests
import fitz
from tqdm.auto import tqdm
import random
import spacy

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def download_pdf(self, url):
        if not os.path.exists(self.pdf_path):
            print("File doest't exist, downloading")
            response = requests.get(url)

            if response.status_code == 200:
                with open(self.pdf_path, "wb") as file:
                    file.write(response.content)
                print(
                    f"The file has been downloaded and saved as: {self.pdf_path}")
            else:
                print(
                    f"Failed to download the file. Status code: {response.status_code}")
        else:
            print(f"File {self.pdf_path} exists.")

    def text_formatter(self, text):
        """Performs minoor formatting on text"""
        clenaed_text = text.replace("\n", " ").strip()
        return clenaed_text

    def open_and_read_pdf(self):
        """Opens a PDF file, reads its text content page by page, and collects statistics"""
        doc = fitz.open(self.pdf_path)
        pages_and_texts = []
        for page_number, page in tqdm(enumerate(doc)):
            text = page.get_text()
            text = self.text_formatter(text)
            pages_and_texts.append({
                "page_number": page_number - 3,  # PDF starts on page 3
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,  # 1 token =~ 4 chars
                "text": text
            })
        return pages_and_texts


    def add_sentences(self, pages_and_texts):
        """Adds sentences to each page using spacy"""
        nlp = spacy.load("en_code_web_sm")
