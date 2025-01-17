# This class generates clusters/voronois in order to achieve faster similarity calculation.
import faiss
from Embedding_Generator import Embedding_Generator
import numpy as np
class Faiss_Voronoi_Generator:
    
    def __init__(self,embeddings_file,N_Cells):
        self.embeddings_file=embeddings_file
        self.embeddings_data=np.load(self.embeddings_file)
        self.N_Cells=N_Cells
        
    def create_voronois(self):
        embeddings_data=self.embeddings_data
        emb_dim=embeddings_data['embeddings'].shape[1]
        quantizer = faiss.IndexFlatL2(emb_dim) 
        index=faiss.IndexIVFFlat(quantizer,emb_dim,self.N_Cells)
        index.train(self.embeddings_data['embeddings']) 
        index.add(embeddings_data['embeddings'])
        return index
