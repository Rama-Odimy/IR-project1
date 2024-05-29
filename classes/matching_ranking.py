import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize
import json

# Function to preprocess the query
def preprocess_query(query):
    # Tokenize and preprocess the query here (e.g., lowercasing, removing stop words, etc.)
    tokens = word_tokenize(query.lower())
    # Implement any additional preprocessing if required
    return ' '.join(tokens)

# Function to compute the cosine similarity and retrieve top document indices
def matching(doc_vector, query_vector):
    # Compute the cosine similarity between the document and query vectors
    cosine_similarities = cosine_similarity(doc_vector, query_vector).flatten()
    
    # Get the indices of the top 10 most similar documents
    related_doc_id = cosine_similarities.argsort()[:-11:-1]
    
    return related_doc_id

# Load the document corpus
with open('D:/ir_final_final_final_the_flinalest/data/antiqe_output/output_collection.tsv', 'r', encoding='utf-8') as f:
    doc_corpus = f.readlines()

# Create and fit the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
doc_vector = tfidf_vectorizer.fit_transform(doc_corpus)

# Save the fitted vectorizer for future use
joblib.dump(tfidf_vectorizer, 'D:/ir_final_final_final_the_flinalest/data/doc_vector_sparse.pkl')

# Example query to be vectorized and matched
query = "sun opening one shade considered insult sun god opening indoors also taboo 11669411"

# Preprocess and vectorize the query using the fitted vectorizer
preprocessed_query = preprocess_query(query)
query_vector = tfidf_vectorizer.transform([preprocessed_query])

# Convert the query vector to a sparse matrix
query_vector = csr_matrix(query_vector)

# Load the saved TF-IDF vectorizer
tfidf_vectorizer = joblib.load('D:/ir_final_final_final_the_flinalest/data/doc_vector_sparse.pkl')

# Vectorize the query using the loaded vectorizer
query_vector = tfidf_vectorizer.transform([preprocessed_query])

# Convert the query vector to a sparse matrix
query_vector = csr_matrix(query_vector)

# Call the matching function
related_doc_id = matching(doc_vector, query_vector)

# Save the related_doc_id in a TXT file
output_file = 'D:/ir_final_final_final_the_flinalest/data/related_doc_id.txt'


print(f"Related document indices saved to {output_file}.")
query = "sun opening one shade considered insult sun god opening indoors also taboo 11669411"

# Preprocess and vectorize the query using the fitted vectorizer
preprocessed_query = preprocess_query(query)
query_vector = tfidf_vectorizer.transform([preprocessed_query])

# Convert the query vector to a sparse matrix
query_vector = csr_matrix(query_vector)

# Load the saved TF-IDF vectorizer
tfidf_vectorizer = joblib.load('D:/ir_final_final_final_the_flinalest/data/doc_vector_sparse.pkl')

# Vectorize the query using the loaded vectorizer
query_vector = tfidf_vectorizer.transform([preprocessed_query])

# Convert the query vector to a sparse matrix
query_vector = csr_matrix(query_vector)

# Call the matching function
related_doc_id = matching(doc_vector, query_vector)

# Save the related_doc_id in a TXT file
output_file = 'D:/ir_final_final_final_the_flinalest/data/related_doc_id.txt'

# Open the file in append mode
with open(output_file, 'a') as f:
    for index in related_doc_id:
        with open('D:/ir_final_final_final_the_flinalest/data/antiqe_output/output_collection.tsv', 'r', encoding='utf-8') as doc_file:
            document_content = doc_file.readlines()[index]
        f.write(f"Document at index {index}:\n")
        f.write(document_content)
        f.write("-----------\n")

print(f"Related document indices saved to {output_file}.")