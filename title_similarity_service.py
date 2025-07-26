from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast

class SimilarityRequest(BaseModel):
    job_title: str
    candidate_titles: list

class SkillSimilarityRequest(BaseModel):
    job_skills: list
    candidate_skills: list

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

@app.post("/skill-similarity")
async def skill_similarity(data: SkillSimilarityRequest):
    # Embed the job skills and candidate skills
    job_skill_embeddings = model.encode(data.job_skills, convert_to_tensor=True)
    candidate_skill_embeddings = model.encode(data.candidate_skills, convert_to_tensor=True)

    # Calculate similarity scores for each skill
    similarity_scores = util.cos_sim(job_skill_embeddings, candidate_skill_embeddings).tolist()

    # Find the best matching skill for each job skill
    best_matches = []
    for i, job_skill in enumerate(data.job_skills):
        candidate_similarities = similarity_scores[i]
        max_score = max(candidate_similarities)
        best_match = data.candidate_skills[candidate_similarities.index(max_score)]
        best_matches.append({
            "job_skill": job_skill,
            "best_match": best_match,
            "max_score": max_score
        })

    return {
        "best_matches": best_matches
    }
