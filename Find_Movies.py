
class Find_Movies:
    def __init__(self,faiss_Vor_index,user_embeddings,top_k):
        # Add the clustered index
        self.faiss_Vor_index=faiss_Vor_index
        self.user_embeddings=user_embeddings
        # Number of the top similar movies 
        self.top_k=top_k
    def find_most_similar_sbert(self):
        faiss_index=self.faiss_Vor_index
        

        # Search for the top_k most similar items
        D, I = faiss_index.search(self.user_embeddings, self.top_k)
    
        return I
    

    





