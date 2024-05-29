import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from scipy.sparse import csr_matrix 

class DataRepresentation:
    def __init__(self, data_file):
        # Initializes the DataRepresentation object.
        # Args:
            # data_file (str): Path to the CSV file containing processed text data.
        self.data_file = data_file
        self.df = pd.read_csv(data_file)
        self.vectorizer = TfidfVectorizer()  # Create a TfidfVectorizer instance
        self.vsm = self.vectorizer.fit_transform(self.df['tokens'])  # Calculate VSM for the dataset
        self.vocabulary = self.vectorizer.get_feature_names_out()  # Store the vocabulary

    def create_vsm(self):
        # Creates a VSM (Vector Space Model) representation of the text data.
        # Returns:
            # scipy.sparse.csr_matrix: A sparse matrix representing the VSM.
        return self.vsm  # Return the pre-calculated VSM

    def get_tfidf_vectors(self):
        # Calculates TF-IDF vectors for the text data.
        # Returns:
            # scipy.sparse.csr_matrix: A sparse matrix of TF-IDF vectors.
        return self.vsm  # Return the pre-calculated VSM

    def get_vocabulary(self):
        # Returns the vocabulary used for the VSM.
        # Returns:
            # list: A list of unique words in the vocabulary.
        return self.vocabulary

    def save_results(self, output_file):
        # Saves the TF-IDF vectors and vocabulary to a binary file.
        # Args:
            # output_file (str): Path to the output file.

        # Convert sparse matrix to dense array
        tfidf_vectors_dense = self.get_tfidf_vectors().toarray()

        # Save the data to a binary file using pickle
        with open(output_file, 'wb') as f:
            pickle.dump((tfidf_vectors_dense, self.vocabulary), f)

    def load_results(self, input_file):
        # Loads the TF-IDF vectors and vocabulary from a binary file.
        # Args:
            # input_file (str): Path to the input file.

        with open(input_file, 'rb') as f:
            tfidf_vectors_dense, vocabulary = pickle.load(f)

        # Convert the dense array back to a sparse matrix
        self.vsm = csr_matrix(tfidf_vectors_dense)
        self.vocabulary = vocabulary

# # Example usage:
# data_rep = DataRepresentation('D:/ir_final_final_final_the_flinalest/data/antiqe_output/output_collection.tsv')
# data_rep.create_vsm()  # Calculate TF-IDF vectors

# # Save the results to a binary file
# data_rep.save_results('D:/ir_final_final_final_the_flinalest/data/antiqe_output/output_collection_tfidf_results.bin')

# # Load the results from the binary file
# data_rep.load_results('D:/ir_final_final_final_the_flinalest/data/antiqe_output/output_collection_tfidf_results.bin')