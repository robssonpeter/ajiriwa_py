from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast

class SimilarityRequest(BaseModel):
    job_title: str
    candidate_titles: list

@app.post("/title-similarity")
async def title_similarity(data: SimilarityRequest):
    job_embedding = model.encode(data.job_title, convert_to_tensor=True)
    candidate_embeddings = model.encode(data.candidate_titles, convert_to_tensor=True)

    similarities = util.cos_sim(job_embedding, candidate_embeddings)[0].tolist()
    max_score = max(similarities)
    best_match = data.candidate_titles[similarities.index(max_score)]

    return {
        "best_match": best_match,
        "max_score": max_score,
        "all_scores": dict(zip(data.candidate_titles, similarities))
    }

