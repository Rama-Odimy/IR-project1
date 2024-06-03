from fastapi import FastAPI
from pydantic import BaseModel
import random
from scipy.sparse import csr_matrix
from matching_ranking import Matching_Ranking
import pickle

app = FastAPI()

class SearchRequest(BaseModel):
    datasetName: str
    query: str

@app.get("/search/data/")
def greet(datasetName: str, query: str):
    # Preprocess and vectorize the query using the loaded vectorizer
    if datasetName == "antique":
        # Load the document corpus
        doc_file = 'D:/ir_final_final_final_the_flinalest/data/antiqe_output/output_collection.tsv'
        doc_id_to_line_number = {}
        with open(doc_file, 'r') as f:
            for line_number, line in enumerate(f):
                # Split the line based on the tab delimiter
                parts = line.strip().split('\t')  # Change delimiter here
                if len(parts) == 2:
                    document_id, text = parts
                    doc_id_to_line_number[document_id] = line_number
            else:
                print(f"Warning: Line {line_number + 1} does not have a tab delimiter: {line.strip()}")
        # Create a Matching_Ranking object
        matcher = Matching_Ranking(doc_file)
        preprocessed_query = matcher.preprocess_query(query)
        query_vector = matcher.data_rep.vectorizer.transform([preprocessed_query])

        # Convert the query vector to a sparse matrix
        query_vector = csr_matrix(query_vector)

        doc_vectors = matcher.data_rep.vsm
        # Call the matching function
        related_doc_id, related_documents = matcher.match(query_vector, doc_vectors)

        # Save the results to a file
        matcher.save_results(related_doc_id, related_documents)
        
        
    elif datasetName == "wikir":
         # Load the document corpus
        doc_file = 'D:/ir_final_final_final_the_flinalest/data/wiki_output/output_documents.tsv'
        # Create a Matching_Ranking object
        matcher = Matching_Ranking(doc_file)
        preprocessed_query = matcher.preprocess_query(query)
        query_vector = matcher.data_rep.vectorizer.transform([preprocessed_query])

        # Convert the query vector to a sparse matrix
        query_vector = csr_matrix(query_vector)

        doc_vectors = matcher.data_rep.vsm
        # Call the matching function
        related_doc_id, related_documents = matcher.match(query_vector, doc_vectors)

        # Save the results to a file
        matcher.save_results(related_doc_id, related_documents) 
    else:
        return {"error": "Invalid dataset name. Please use 'antique' or 'wikir'."}
    
    response = {
        "related_documents": related_documents  # Return the related documents
    }

    return response