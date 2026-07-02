# 🧠 Cognitive Sourcing: Beyond Semantic Matching
**India Runs AI Hackathon - Redrob Submission**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Sandbox-red)](https://streamlit.io/)

## 🚀 Overview
Traditional ATS keyword filters are easily gamed by "resume-stuffing" and fail to capture actual technical depth. This project introduces a **Hybrid Candidate Discovery Engine** engineered to process 100,000 candidate profiles locally on CPU in under 5 minutes. 

Instead of relying on fragile string-matching or slow, hallucination-prone generative LLMs, this system combines **Dense Vector Semantic Matching** (to understand engineering context) with a **Deterministic Python Rules Engine** (to aggressively filter out unhirable traits and strictly enforce business logic).

### 🔗 Quick Links
* **Interactive Sandbox (Hugging Face):** [(https://huggingface.co/spaces/aashipandey513-stack/indiaruns-ai-recruiter)]

* **Final Submission CSV:** `submission.csv`

---

## 🌐 The Sandbox Experience
To provide full transparency into our ranking logic, we deployed a live Sandbox on Hugging Face Spaces.
* **Why we built it:** To demonstrate that our heuristic-based scoring is both explainable and deterministic.
* **What it does:** Users can input a Job Description and instantly see how the engine ranks the candidate pool, including:
    * **Semantic Scores:** Showing how "Engineering Context" is derived from vector similarity.
    * **Heuristic Multipliers:** Real-time visibility into how "Consulting Traps" or "Behavioral Boosts" adjust the final ranking.

---

## 🏗️ System Architecture

### Phase 1: Semantic Embedding (The "Brain")
* Evaluates the candidate pool against the target Job Description (JD) using the `all-MiniLM-L6-v2` Sentence Transformer.
* Parses nested JSON (skills, summaries, career histories) into a unified semantic string and calculates Cosine Similarity.
* Accurately surfaces candidates with genuine experience in embeddings, LLMs, and vector databases, regardless of the exact keywords they used.

### Phase 2: Deterministic Rules Engine (The "Guardrails")
Pure vector math can sometimes match adjacent but irrelevant profiles. To guarantee rigorous engineering quality, post-processing heuristic multipliers are applied to the base semantic score:
1. **The "Non-Engineering" Penalty (0.1x Multiplier):** Heavily penalizes candidates holding non-technical titles (e.g., Marketing Manager, Accountant) who stuffed their resumes with AI buzzwords.
2. **The "Consulting Trap" Penalty (0.2x Multiplier):** Penalizes profiles consisting *exclusively* of IT consulting firms to strictly enforce the JD's requirement for internal product-level deployment experience.
3. **Behavioral Intent Boost (0.5x to 1.2x Multiplier):** Leverages `redrob_signals`. Candidates with a `<10%` recruiter response rate are penalized as "ghosts," while highly active candidates (`>70%` response rate) receive a 20% score boost, ensuring the shortlist is highly hirable.

**Final Score Calculation:** 
`Final Score = Base Vector Similarity * Behavioral Multipliers`

---

## 📂 Repository Structure

    indiaruns-ai-recruiter/
    │
    ├── data/
    │   └── candidates.jsonl             # Large dataset (100K profiles) - GitIgnored
    │
    ├── src/
    │   └── generate_submission.py       # Main engine: batch processing & scoring
    │
    ├── README.md                        # Documentation
    ├── submission_metadata.yaml         # Team & architecture configuration
    ├── requirements.txt                 # Dependencies
    └── validate_submission.py           # Official Redrob Output validation script

---

## ⚙️ How to Reproduce (Local Setup)

This system is optimized for high-speed CPU inference. It processes candidates in batches to maintain a low and stable memory footprint.

**1. Clone the repository and navigate to the project directory:**
    git clone [(https://github.com/aashipandey513-stack/indiaruns-ai-recruiter)]
    cd indiaruns-ai-recruiter

**2. Set up the Python virtual environment:**
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate

**3. Install dependencies:**
    pip install -r requirements.txt

**4. Ensure the dataset is in place:**
    Place the unzipped candidates.jsonl file inside the data/ folder.

**5. Ignite the processing engine:**
    python src/generate_submission.py
    
*The script will stream the JSONL file in batches, keeping RAM usage low. Processing 100,000 candidates takes approximately 3-5 minutes on a standard 4-core machine.*

**6. Validate the output:**
    python validate_submission.py submission.csv

---

## 📊 Performance & Constraints
* **Throughput:** ~100,000 profiles processed in under 5 minutes.
* **Compute:** 100% Local CPU processing (No external API latency, rate limits, or GPU requirement).
* **Explainability:** 100% deterministic scoring. Zero hallucination risk. The dynamic string generator outputs a clear, logic-based sentence for every candidate in the `reasoning` column based on their specific semantic score and triggered heuristic multipliers.

---
*Engineered by Aashi Pandey | Solo Participant| Team: HACKALONE*
