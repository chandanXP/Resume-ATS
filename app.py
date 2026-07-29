import re
import nltk
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK resources are available silently
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = FastAPI(title="Smart ATS Resume Copilot")

# Sample Skill Database for Information Extraction (NER)
SKILL_DB = [
    "python", "javascript", "typescript", "react", "node.js", "express", "mongodb",
    "sql", "postgresql", "fastapi", "docker", "aws", "git", "machine learning",
    "nlp", "tensorflow", "pytorch", "pandas", "numpy", "html", "css", "tailwind"
]

def clean_and_tokenize(text: str):
    """Tokenization and Stop Word Removal Pipeline"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return filtered_tokens

# ==========================================
# 🟢 CODE CHANGES: CORE ATS ENGINE LOGIC
# ==========================================
def extract_entities(text: str):
    """Extracts candidate contact details and skills (NER)"""
    # Extract Email using Regex
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    email = emails[0] if emails else "Not Found"

    # Extract Phone Number using Regex
    phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    phone = phones[0] if phones else "Not Found"

    # Extract Skills via Token Matching
    text_lower = text.lower()
    found_skills = [skill for skill in SKILL_DB if skill in text_lower]

    return {
        "email": email,
        "phone": phone,
        "skills": list(set(found_skills))
    }

def calculate_match_score(resume_text: str, job_text: str):
    """Calculates Match Percentage using TF-IDF and Cosine Similarity"""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    
    # Cosine Similarity between vector 0 (Resume) and vector 1 (Job Description)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_percentage = round(similarity * 100, 2)
    return match_percentage

def find_missing_keywords(resume_text: str, job_text: str):
    """Identifies missing skills required by the job description"""
    job_skills = [skill for skill in SKILL_DB if skill in job_text.lower()]
    resume_skills = [skill for skill in SKILL_DB if skill in resume_text.lower()]
    
    missing = set(job_skills) - set(resume_skills)
    return list(missing)
# ==========================================

# --- Web UI Endpoint ---
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Resume & ATS Copilot</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-white min-h-screen p-8">
        <div class="max-w-4xl mx-auto">
            <h1 class="text-3xl font-bold mb-2 text-cyan-400">🤖 Smart ATS Resume Copilot</h1>
            <p class="text-slate-400 mb-8">Analyze candidate resumes against job descriptions using NLP, TF-IDF & Similarity Matching.</p>
            
            <form action="/analyze" method="post" class="space-y-6">
                <div>
                    <label class="block text-sm font-medium mb-2 text-slate-300">Paste Candidate Resume Text</label>
                    <textarea name="resume_text" rows="6" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-500" required></textarea>
                </div>
                <div>
                    <label class="block text-sm font-medium mb-2 text-slate-300">Paste Job Description</label>
                    <textarea name="job_text" rows="6" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-500" required></textarea>
                </div>
                <button type="submit" class="w-full bg-cyan-500 hover:bg-cyan-600 font-semibold py-3 px-6 rounded-lg transition duration-200">Run ATS Analysis</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/analyze", response_class=HTMLResponse)
def analyze(resume_text: str = Form(...), job_text: str = Form(...)):
    entities = extract_entities(resume_text)
    score = calculate_match_score(resume_text, job_text)
    missing_skills = find_missing_keywords(resume_text, job_text)

    skills_badge = "".join([f'<span class="bg-cyan-900 text-cyan-300 px-3 py-1 rounded-full text-xs font-semibold mr-2">{s}</span>' for s in entities["skills"]])
    missing_badge = "".join([f'<span class="bg-rose-900 text-rose-300 px-3 py-1 rounded-full text-xs font-semibold mr-2">{s}</span>' for s in missing_skills])

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ATS Results</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-white min-h-screen p-8">
        <div class="max-w-4xl mx-auto space-y-6">
            <a href="/" class="text-cyan-400 hover:underline">← Analyze Another Resume</a>
            
            <div class="bg-slate-800 border border-slate-700 rounded-lg p-6">
                <h2 class="text-2xl font-bold text-cyan-400 mb-4">ATS Match Score: {score}%</h2>
                <div class="w-full bg-slate-700 h-4 rounded-full overflow-hidden">
                    <div class="bg-cyan-500 h-full" style="width: {score}%"></div>
                </div>
            </div>

            <div class="grid grid-cols-2 gap-6">
                <div class="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-3">
                    <h3 class="text-lg font-bold text-slate-200">Candidate Information</h3>
                    <p class="text-sm text-slate-400"><strong>Email:</strong> {entities['email']}</p>
                    <p class="text-sm text-slate-400"><strong>Phone:</strong> {entities['phone']}</p>
                    <div class="pt-2">
                        <p class="text-sm text-slate-400 mb-2"><strong>Extracted Skills:</strong></p>
                        <div class="flex flex-wrap gap-2">{skills_badge or '<span class="text-slate-500">None detected</span>'}</div>
                    </div>
                </div>

                <div class="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-3">
                    <h3 class="text-lg font-bold text-rose-400">Missing Required Skills</h3>
                    <p class="text-xs text-slate-400">Skills present in Job Description but missing from Resume:</p>
                    <div class="flex flex-wrap gap-2 pt-2">{missing_badge or '<span class="text-emerald-400 text-sm">None! Candidate covers all required skills.</span>'}</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
