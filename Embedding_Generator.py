# This file is not directly used in the app but needs to be run at the beggining for once.
# Running this file is needed to generate Sbert embeddings and save it as .npz file so it can be used in other modules.

import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd




    
class Embedding_Generator:
    #Initialize the generator
    def __init__(self,data_file) :
       self.data_file=data_file
       self.model=SentenceTransformer('all-mpnet-base-v2') 
       #Retrieve data from file
       self.data=self.load_movie_data() 
          
       
    def load_movie_data(self): 


        
        #Read data
        df = pd.read_csv(self.data_file)
        #Load the collumns
        titles = df['title'].tolist() 
        overviews = df['overview'].tolist()  
        keywords=df['keywords'].tolist()
        indices = df['index'].tolist()    
        return indices,titles,overviews,keywords




    def get_sbert_embeddings(self):
        #Generated Embedding will be stored in this list
        movie_embeddings = [] 
        # Load data
        indices,titles,overviews,keywords=self.data
        # Generate the embeddings all along the list
        for i in range(len(indices)):
            # Check if both are not empty
            if overviews[i] and keywords[i]: 
                #Combine the overviews & keywords.[SEP] is neccesary so the model can interpret them separately 
                combined_input = f"{overviews[i]} [SEP] {keywords[i]}"
                # Sbert encoding
                embedding = self.model.encode(combined_input)
                # Append the generated embedding to the embeddings list
                movie_embeddings.append(embedding)
               #Return List of embeddings   
        return np.array(movie_embeddings) 
    

def main():
    #Initialize the Generator
    data_file='movies_with_keywords.csv'
    Generator=Embedding_Generator(data_file)
    
  
    
    # Generate embeddings 
    all_embeddings = Generator.get_sbert_embeddings
    #Load data again to add it in .npz file
    indices,titles,overviews,keywords=Generator.data
    # Save embeddings in .npz for later use
    np.savez("sbert_embeddings.npz", embeddings=all_embeddings,indices=np.array(indices),titles=titles,overviews=overviews,keywords=keywords )
    print('generated sentence embeddings:',all_embeddings)
    print('number of embeddings:',len(all_embeddings))
if __name__ == "__main__":
    main()   







    

