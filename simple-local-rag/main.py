import os
import requests
import fitz
from tqdm.auto import tqdm
import random
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer


class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def download_pdf(self, url):
        
        if not os.path.exists(self.pdf_path):
            print("File doesn't exist, downloading...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(self.pdf_path, "wb") as file:
                    file.write(response.content)
                print(f"The file has been downloaded and saved as: {self.pdf_path}")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")
        else:
            print(f"File {self.pdf_path} exists.")

    def text_formatter(self, text):
        """Performs minor formatting on text"""
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text

    def open_and_read_pdf(self):
        """Opens a PDF file, reads its text content page by page, and collects statistics"""
        doc = fitz.open(self.pdf_path)
        pages_and_texts = []
        for page_number, page in tqdm(enumerate(doc)):
            text = page.get_text()
            text = self.text_formatter(text)
            pages_and_texts.append({
                "page_number": page_number - 41,  # PDF starts on page 3
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,  # 1 token =~ 4 chars
                "text": text
            })
        return pages_and_texts

    def add_sentences(self, pages_and_texts):
        """Adds sentences to each page using spacy"""
        nlp = English()
        nlp.add_pipe("sentencizer")        
        for item in pages_and_text:
            item["sentences"] = list(nlp(item["text"]).sents)
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            item["page_sentence_count_spacy"] = len(item["sentences"])
            
    def split_list(self, input_list, slice_size):
        """Splits the input_list into sublists of size slice_size"""
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


    def split_sentences_into_chunks(self, pages_and_text, num_sentence_chunk_size):
        """Splits sentences into chunks for each page."""
        for item in pages_and_text:
            item["sentence_chunks"] = self.split_list(input_list=item["sentences"], 
                                                      slice_size=num_sentence_chunk_size)
            item["num_chunks"] = len(item["sentence_chunks"])
            
    def chunks_into_items(self, pages_and_text):
        pages_and_chunks = []
        for item in pages_and_text:
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                # Join the sentences together into a paragraph-like structure, aka a chunk
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter
    
              # Get stats about chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token ~= 4 characters
                pages_and_chunks.append(chunk_dict)
    
        return pages_and_chunks
    
    def embedding(self):
        embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                             device="cpu")
        sentences = ["Hello World",
                    "Am Kajetan Frackowiak",
                    "How's your life?"]
        embeddings = embedding_model.encode(sentences)
        embeddings_dict = dict(zip(sentences, embeddings))
        
        print(f"Embedding size: {embeddings_dict[1].shape}")

    
# Constants
PDF_PATH = "human-nutrition-text.pdf"
PDF_URL = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

import pandas as pd

# Main functionality
# Ensure that 'chunk_token_count' is added to DataFrame correctly
pdf_processor = PDFProcessor(pdf_path=PDF_PATH)
pdf_processor.download_pdf(url=PDF_URL)
pages_and_text = pdf_processor.open_and_read_pdf()
pdf_processor.add_sentences(pages_and_text)
num_sentence_chunk_size = 10
pdf_processor.split_sentences_into_chunks(pages_and_text, num_sentence_chunk_size)
pages_and_chunks = pdf_processor.chunks_into_items(pages_and_text)
df = pd.DataFrame(pages_and_chunks)

# Address KeyError
if 'chunk_token_count' not in df.columns:
    print("Error: 'chunk_token_count' column not found in DataFrame.")
else:
    # Sample rows from DataFrame
    min_token_length = 30
    if len(df) >= 5:
        sampled_df = df[df["chunk_token_count"] <= min_token_length].sample(5, replace=True)
        for _, row in sampled_df.iterrows():
            print(f'Chunk token count: {row["chunk_token_count"]} | Text: {row["text"]}')
    else:
        print("Error: DataFrame has fewer than 5 rows.")
