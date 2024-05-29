from fastapi import FastAPI
from pydantic import BaseModel
import random
from query_processing import process_query
from matching_ranking import matching

app = FastAPI()

class SearchRequest(BaseModel):
    datasetName: str
    query: str

@app.get("/search")
def greet(datasetName: str, query: str):
    output_file, query_vector = process_query(query, dataset_name='antique')
    with open(output_file, 'r') as file:  # Use the correct output_file
        file_content = file.read()
    matcher = matching()

    # Call the match function with the query vector
    related_doc_id = matcher.match(query_vector)

    # Save the results to a file
    matcher.save_results(related_doc_id)
    

    # Return the file content in the response
    return {"message": f"Hello, You are searching in {datasetName} about {query}!",
            "results": file_content} 
