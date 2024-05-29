from text_processing import TextProcessing
from representation_data import DataRepresentation  # Import DataRepresentation

def process_query(user_query, output_file='query_output.tsv', dataset_name='antique'):
    # Processes a user query using the TextProcessing class and generates a TF-IDF vector.
    # Args:
        # user_query (str): The user's query string.
        # output_file (str, optional): The path to the output TSV file. Defaults to 'query_output.tsv'.
        # dataset_name (str, optional): The name of the dataset to use for vectorization. Defaults to 'antique'.
    # Returns:
        # tuple: A tuple containing the path to the output file and the TF-IDF vector of the query.
    
    processor = TextProcessing()
    df = processor.process_text(user_query)
    df.to_csv(output_file, sep='\t', index=False)

    # Create DataRepresentation object for the processed query
    data_rep = DataRepresentation(output_file)  # Use the output file as input
    data_rep.create_vsm()  # Calculate TF-IDF vectors

    # Get the TF-IDF vector for the query
    query_vector = data_rep.get_tfidf_vectors()

    # Save the query vector to a file in the specified path
    output_path = 'D:/ir_final_final_final_the_flinalest/data/antiqe_output'
    with open(f'{output_path}/query_tfidf_results.tsv', 'w') as f:
        for i, value in enumerate(query_vector.toarray()[0]):
            f.write(f"{data_rep.vocabulary[i]}\t{value}\n")

    return output_file, query_vector

# # Example usage:
# output_file, query_vector = process_query("Query antique", dataset_name='antique')
# print(f"Output file: {output_file}")
# print(f"Query vector: {query_vector}")