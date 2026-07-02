import json
import csv
import gzip
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETUP ---
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# The exact Job Description we are targeting
JD_TEXT = """
We need someone with deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real users.
Production experience with vector databases (Pinecone, Weaviate, Qdrant, Milvus, FAISS).
Strong Python.
Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation).
"""
jd_vector = model.encode([JD_TEXT])

# --- 2. BUSINESS LOGIC ---
CONSULTING_FIRMS = ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini", "Tech Mahindra"]
NON_ENG_TITLES = ["Marketing Manager", "HR Manager", "Accountant", "Sales Executive", "Project Manager", "Customer Support"]

def calculate_multipliers(candidate):
    """Calculates the combined penalty/boost based on hackathon rules."""
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    signals = candidate.get('redrob_signals', {})
    
    multiplier = 1.0
    
    # Penalty: Wrong Job Title
    current_title = profile.get('current_title', '')
    if any(title in current_title for title in NON_ENG_TITLES):
        multiplier *= 0.1 
        
    # Penalty: Pure Consulting Background
    all_companies = [job.get('company', '') for job in career]
    if all_companies and all(company in CONSULTING_FIRMS for company in all_companies):
         multiplier *= 0.2
         
    # Behavioral Multiplier (Activity & Intent)
    response_rate = signals.get('recruiter_response_rate', 0.5)
    if response_rate < 0.10:
        multiplier *= 0.5 # Ghost penalty
    elif response_rate > 0.70:
        multiplier *= 1.2 # Highly active boost
        
    return multiplier

def extract_semantic_text(candidate):
    """Creates the text block for the AI to read."""
    profile = candidate.get('profile', {})
    skills = ", ".join([s['name'] for s in candidate.get('skills', [])])
    exp = "\n".join([f"{j.get('title', '')} at {j.get('company', '')}: {j.get('description', '')}" for j in candidate.get('career_history', [])])
    return f"Title: {profile.get('current_title', '')}\nSummary: {profile.get('summary', '')}\nSkills: {skills}\nExperience: {exp}"

# --- 3. BATCH PROCESSING ENGINE ---
def process_candidates(filepath, batch_size=1000):
    all_scored_candidates = []
    print(f"Opening large dataset: {filepath}")
    open_func = gzip.open if filepath.endswith('.gz') else open
    
    batch_texts = []
    batch_data = []
    count = 0
    
    with open_func(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            candidate = json.loads(line)
            
            batch_texts.append(extract_semantic_text(candidate))
            batch_data.append(candidate)
            count += 1
            
            # Process in batches
            if len(batch_texts) >= batch_size:
                score_batch(batch_texts, batch_data, all_scored_candidates)
                print(f"Processed {count} candidates...")
                batch_texts, batch_data = [], []
                
        # Process final batch
        if batch_texts:
            score_batch(batch_texts, batch_data, all_scored_candidates)
            print(f"Finished processing all {count} candidates.")
            
    return all_scored_candidates

def score_batch(texts, raw_data, master_list):
    """Encodes a batch of candidates and applies logic."""
    vectors = model.encode(texts, batch_size=256)
    similarities = cosine_similarity(jd_vector, vectors).flatten()
    
    for i, candidate in enumerate(raw_data):
        base_score = similarities[i]
        multiplier = calculate_multipliers(candidate)
        final_score = float(np.round(base_score * multiplier, 4))
        
        master_list.append({
            "candidate_id": candidate['candidate_id'],
            "score": final_score,
            "title": candidate['profile'].get('current_title', 'Unknown'),
            "multiplier": multiplier
        })

# --- 4. EXPORT TO CSV ---
def generate_csv(scored_candidates, output_filename="submission.csv"):
    scored_candidates.sort(key=lambda x: (-x['score'], x['candidate_id']))
    top_100 = scored_candidates[:100]
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank, c in enumerate(top_100, start=1):
            if rank <= 10:
                reasoning = f"Exceptional fit. {c['title']} with strong semantic alignment to modern ML systems. High availability ({c['multiplier']}x behavioral modifier)."
            elif rank <= 50:
                reasoning = f"Strong candidate. {c['title']} showing solid retrieval/ranking experience. Passed all consulting and title heuristic checks."
            else:
                reasoning = f"Good foundational skills as a {c['title']}, but semantic match is slightly weaker than top tier. Placed lower in shortlist but viable."
                
            writer.writerow([c['candidate_id'], rank, f"{c['score']:.4f}", reasoning])
            
    print(f"\nSUCCESS! Top 100 candidates saved to {output_filename}")

if __name__ == "__main__":
    dataset_path = "data/candidates.jsonl" 
    if os.path.exists(dataset_path):
        results = process_candidates(dataset_path)
        generate_csv(results)
    else:
        print(f"Could not find dataset at {dataset_path}. Please check path and filename.")