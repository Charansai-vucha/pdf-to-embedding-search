# main.py

# Step 1: Install Required Libraries
!pip install pymupdf sentence-transformers faiss-cpu

# Step 2: Extract Content from PDF
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Open the PDF
    text = ""
    for page_num in range(len(doc)):  # Loop through each page
        page = doc.load_page(page_num)  # Load the page
        text += page.get_text()  # Extract text
    return text  # Return concatenated text

# Example usage
pdf_path = 'path_to_your_pdf.pdf'  # Change to the actual path
pdf_text = extract_text_from_pdf(pdf_path)  # Extract text
print(pdf_text)  # Print extracted text

# Step 3: Create Embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

sentences = pdf_text.split('\n')  # Split the text into sentences based on newlines

embeddings = model.encode(sentences)  # Create embeddings for each sentence
print(embeddings.shape)  # Print the shape of the embeddings array

# Step 4: Store Embeddings in Vector Store (FAISS)
import faiss
import numpy as np

# Create a FAISS index for similarity search
d = embeddings.shape[1]  # Get the dimensionality of embeddings
index = faiss.IndexFlatL2(d)  # Create an index using L2 distance
index.add(np.array(embeddings))  # Add embeddings to the index
print(f"Number of vectors in the index: {index.ntotal}")  # Print the number of vectors

# Step 5: Perform Similarity Search
def search_similar(query, top_k=5):
    query_embedding = model.encode([query])  # Create embedding for the query
    D, I = index.search(np.array(query_embedding), top_k)  # Search for top_k similar sentences
    return I[0], D[0]  # Return the indices and distances

# Example search
query = "your search query here"  # Change this to your search query
top_k = 5  # Get top 5 similar sentences
indices, distances = search_similar(query, top_k)  # Perform similarity search

# Print the top similar sentences
print("Top similar sentences:")
for idx, dist in zip(indices, distances):
    print(f"Sentence: {sentences[idx]}, Distance: {dist}")
