"""
INSIGNIA — AI Interview Prep Suite
FastAPI Backend with Google Gemini LLM
"""

import os
import sys
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# -- Load env ------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

client = None
if GENAI_AVAILABLE and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("[Insignia] Gemini AI client initialized.")
    except Exception as e:
        print(f"[Insignia] Gemini init failed: {e}")
else:
    print("[Insignia] GEMINI_API_KEY not set -- LLM features will use fallback templates.")
    print("[Insignia] Create backend/.env with: GEMINI_API_KEY=your_key_here")

# ── Supabase ──────────────────────────────────────────────────────────────────
supabase_db = None
try:
    from supabase import create_client, Client
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qzhodtpzajupwgoghmcw.supabase.co")
    SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF6aG9kdHB6YWp1cHdnb2dobWN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY0NDQ4NzksImV4cCI6MjA5MjAyMDg3OX0.av43zZ-nAZg4FOnHfXNRK_LKL-CAeUZR-ewgdn3VbsI")
    supabase_db = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[Insignia] Supabase Python Client connected.")
except Exception as e:
    print(f"[Insignia] Supabase Python Client not available: {e}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Insignia AI Backend",
    description="Powers resume generation, Q&A, study plans, and mock interview feedback using Google Gemini.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

def ask_gemini(prompt: str, fallback: str = "") -> str:
    """Call Gemini and return text, falling back gracefully."""
    if not client:
        return fallback
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return fallback

# ═══════════════════════════════════════════════════════════════════════════════
#  RESUME BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class EducationItem(BaseModel):
    degree: str
    institution: str
    year: Optional[str] = ""
    gpa: Optional[str] = ""

class ExperienceItem(BaseModel):
    title: str
    company: str
    duration: Optional[str] = ""
    desc: Optional[str] = ""

class ProjectItem(BaseModel):
    name: str
    tech: Optional[str] = ""
    desc: Optional[str] = ""

class ResumeRequest(BaseModel):
    name: str
    email: str
    phone: Optional[str] = ""
    location: Optional[str] = ""
    linkedin: Optional[str] = ""
    portfolio: Optional[str] = ""
    targetRole: str
    level: str
    summary: Optional[str] = ""
    skills: str
    education: List[EducationItem] = []
    experience: List[ExperienceItem] = []
    projects: List[ProjectItem] = []

class ResumeResponse(BaseModel):
    enhanced_summary: str
    experience_bullets: List[dict]
    key_achievements: List[dict]
    suggestions: List[str]


@app.post("/api/resume/generate", response_model=ResumeResponse)
async def generate_resume(req: ResumeRequest):
    """
    Takes resume form data → returns LLM-enhanced content:
    - Professional summary
    - Improved experience bullet points
    - Key achievements extracted
    - Improvement suggestions
    """

    # ── Build context string for Gemini ───────────────────────────────────────
    exp_text = ""
    for ex in req.experience:
        exp_text += f"\n- {ex.title} at {ex.company} ({ex.duration}): {ex.desc}"

    proj_text = ""
    for p in req.projects:
        proj_text += f"\n- {p.name} ({p.tech}): {p.desc}"

    edu_text = ""
    for e in req.education:
        edu_text += f"\n- {e.degree} from {e.institution} ({e.year})"

    # ── Summary prompt ────────────────────────────────────────────────────────
    summary_prompt = f"""
You are an expert resume writer. Write a compelling, ATS-optimized professional summary for:

Name: {req.name}
Target Role: {req.targetRole}
Experience Level: {req.level}
Skills: {req.skills}
Experience:{exp_text if exp_text else " No prior experience (fresher)"}
User's draft summary: "{req.summary or 'none provided'}"

Rules:
- 3-4 sentences maximum
- Start with level adjective (e.g., "Seasoned", "Results-driven")
- Include 2-3 key skills naturally
- End with value proposition for employer
- No generic phrases like "hardworking team player"
- Plain text only, no markdown

Return ONLY the summary paragraph, nothing else.
"""

    enhanced_summary = ask_gemini(
        summary_prompt,
        fallback=f"Results-driven {req.targetRole} with expertise in {', '.join(req.skills.split(',')[:3])}. "
                 f"Passionate about delivering high-quality solutions and driving measurable impact through "
                 f"technical excellence and collaborative leadership."
    )

    # ── Experience bullets prompt ─────────────────────────────────────────────
    enhanced_experience = []
    for ex in req.experience:
        if not ex.desc:
            enhanced_experience.append({"title": ex.title, "company": ex.company,
                                        "duration": ex.duration, "bullets": []})
            continue

        bullets_prompt = f"""
Rewrite these job responsibilities as 4-6 powerful resume bullet points for a {req.targetRole} role:

Job: {ex.title} at {ex.company}
Original description: {ex.desc}

Rules:
- Start each bullet with a strong action verb (Architected, Optimized, Led, Delivered, etc.)
- Quantify results where possible (use % improvements, numbers, scale)
- Focus on impact and achievements, not just duties
- Each bullet max 25 words
- Return ONLY a JSON array of strings like: ["bullet 1", "bullet 2", ...]
- No markdown, no explanation, just the JSON array
"""
        raw = ask_gemini(bullets_prompt, fallback="")
        bullets = []
        if raw:
            try:
                # Strip markdown code fences if present
                clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                bullets = json.loads(clean)
            except Exception:
                # Fallback: split by newlines
                bullets = [l.strip().lstrip("-•* ") for l in raw.split("\n") if l.strip()]
        else:
            bullets = [l.strip().lstrip("-•* ") for l in ex.desc.split("\n") if l.strip()]

        enhanced_experience.append({
            "title": ex.title,
            "company": ex.company,
            "duration": ex.duration,
            "bullets": bullets
        })

    # ── Key Achievements ──────────────────────────────────────────────────────
    achievements_prompt = f"""
Based on this professional profile, generate 4 impactful key achievements:

Role: {req.targetRole} ({req.level})
Experience:{exp_text if exp_text else " Fresher with strong foundational skills"}
Projects:{proj_text if proj_text else " None listed"}
Skills: {req.skills}

Rules:
- Each achievement should have a short title (4-6 words) and a 1-2 sentence description
- Focus on metrics, innovation, and impact
- Be specific and credible
- Return ONLY a JSON array: [{{"title": "...", "desc": "..."}}, ...]
- No markdown, no explanation
"""
    raw_achievements = ask_gemini(achievements_prompt, fallback="")
    key_achievements = []
    if raw_achievements:
        try:
            clean = raw_achievements.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            key_achievements = json.loads(clean)
        except Exception:
            pass

    if not key_achievements:
        key_achievements = [
            {"title": "Driving Technical Excellence", "desc": f"Consistently delivering high-quality {req.targetRole} solutions that exceed project requirements and stakeholder expectations."},
            {"title": "Cross-functional Collaboration", "desc": "Successfully partnering with diverse teams to align technical solutions with business objectives and drive organizational growth."},
            {"title": "Continuous Skill Development", "desc": f"Proactively expanding expertise in {req.skills.split(',')[0].strip()} and emerging technologies to stay ahead of industry trends."},
            {"title": "Results-Oriented Delivery", "desc": "Track record of completing projects on schedule while maintaining high standards of code quality and system reliability."}
        ]

    # ── Suggestions ───────────────────────────────────────────────────────────
    suggestions_prompt = f"""
Review this resume profile and give 5 specific, actionable improvement suggestions:

Name: {req.name}
Role: {req.targetRole} ({req.level})
Skills listed: {req.skills}
Has experience: {bool(req.experience)}
Has LinkedIn: {bool(req.linkedin)}
Has portfolio: {bool(req.portfolio)}
Number of skills: {len(req.skills.split(','))}

Return ONLY a JSON array of 5 suggestion strings.
Each suggestion should be specific, actionable, and max 20 words.
No markdown, no explanation.
"""
    raw_suggestions = ask_gemini(suggestions_prompt, fallback="")
    suggestions = []
    if raw_suggestions:
        try:
            clean = raw_suggestions.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            suggestions = json.loads(clean)
        except Exception:
            suggestions = [l.strip().lstrip("-•*0123456789. ") for l in raw_suggestions.split("\n") if l.strip()]

    if len(suggestions) < 3:
        suggestions.extend([
            "Add 2-3 quantified achievements with specific metrics (%, $, time saved).",
            "Include a GitHub or portfolio link — 73% of recruiters check candidate portfolios.",
            "Tailor your skills section to match keywords in the job description for ATS optimization.",
            "Use the STAR method in experience descriptions: Situation, Task, Action, Result.",
            "Keep resume to one page for under 5 years experience; two pages max for seniors."
        ])

    if supabase_db and req.email:
        try:
            supabase_db.table("resume_history").insert({
                "email": req.email,
                "target_role": req.targetRole,
                "level": req.level,
                "summary_generated": enhanced_summary
            }).execute()
        except Exception as e:
            print(f"[Insignia] Failed to push resume to Supabase: {e}")

    return ResumeResponse(
        enhanced_summary=enhanced_summary,
        experience_bullets=enhanced_experience,
        key_achievements=key_achievements,
        suggestions=suggestions[:6]
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Q&A GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class QARequest(BaseModel):
    role: str
    level: str
    type: str      # "technical" | "behavioral" | "situational" | "all"
    count: int = 10

class QAItem(BaseModel):
    question: str
    model_answer: str
    type: str
    criteria: str

@app.post("/api/qa/generate", response_model=List[QAItem])
async def generate_qa(req: QARequest):
    """Generate role-specific interview Q&A using Gemini."""

    type_map = {
        "technical": "technical and coding questions",
        "behavioral": "behavioral STAR-method questions",
        "situational": "situational judgment questions",
        "all": "a mix of technical, behavioral, and situational questions"
    }
    q_type_desc = type_map.get(req.type, "mixed questions")

    prompt = f"""
Generate exactly {req.count} interview questions with model answers for:

Role: {req.role}
Experience Level: {req.level}
Question Types: {q_type_desc}

Return ONLY a valid JSON array. Each item must have exactly these fields:
- "question": the interview question (string)
- "model_answer": a detailed model answer (2-4 sentences) (string)
- "type": one of "technical", "behavioral", or "situational" (string)
- "criteria": what the interviewer is looking for (1 sentence) (string)

Format: [{{"question": "...", "model_answer": "...", "type": "...", "criteria": "..."}}, ...]

No markdown, no code fences, no explanation. Return RAW JSON only.
"""

    raw = ask_gemini(prompt, fallback="")
    questions = []

    if raw:
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            for item in data:
                questions.append(QAItem(
                    question=item.get("question", ""),
                    model_answer=item.get("model_answer", ""),
                    type=item.get("type", req.type if req.type != "all" else "technical"),
                    criteria=item.get("criteria", "Understanding and communication skills.")
                ))
        except Exception as e:
            print(f"Q&A parse error: {e}")

    if not questions:
        # Fallback questions
        questions = [
            QAItem(question=f"Tell me about your experience with {req.role} responsibilities.",
                   model_answer="Use the STAR method: describe a specific situation, the task you were assigned, the actions you took, and the measurable results you achieved.",
                   type="behavioral", criteria="Clarity, structure, and relevance of experience."),
            QAItem(question=f"What are the most important technical skills for a {req.role}?",
                   model_answer=f"A {req.role} should have strong foundations in core domain skills, ability to learn new technologies quickly, and practical experience applying them in real-world scenarios.",
                   type="technical", criteria="Depth of technical knowledge and self-awareness."),
            QAItem(question="How do you handle a situation where you disagree with your manager's technical decision?",
                   model_answer="I raise my concerns professionally with data and reasoning, propose alternatives, and ultimately respect the team's decision while documenting my perspective.",
                   type="situational", criteria="Professionalism, communication, and teamwork."),
        ]

    return questions[:req.count]


# ═══════════════════════════════════════════════════════════════════════════════
#  STUDY PLAN
# ═══════════════════════════════════════════════════════════════════════════════

class StudyPlanRequest(BaseModel):
    domain: str
    level: str   # "beginner" | "intermediate" | "advanced"

class StudyTopic(BaseModel):
    name: str
    description: str
    key_points: List[str]
    resources: List[str]

class StudyModule(BaseModel):
    title: str
    topics: List[StudyTopic]

@app.post("/api/study/generate", response_model=List[StudyModule])
async def generate_study_plan(req: StudyPlanRequest):
    """Generate a structured study plan using Gemini."""

    prompt = f"""
Create a comprehensive study plan for:

Domain/Role: {req.domain}
Level: {req.level}

Return EXACTLY 4 study modules as a JSON array. Each module has:
- "title": Module title (string)
- "topics": Array of 2-3 topics, each with:
  - "name": Topic name (string)
  - "description": 1-sentence description (string)
  - "key_points": Array of 3-4 specific things to learn (strings)
  - "resources": Array of 2 resource suggestions like "Book: ...", "Course: ..." (strings)

Return RAW JSON only. No markdown, no code fences, no explanation.
Format: [{{"title": "...", "topics": [{{"name": "...", "description": "...", "key_points": [...], "resources": [...]}}]}}]
"""

    raw = ask_gemini(prompt, fallback="")
    modules = []

    if raw:
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            for mod in data:
                topics = []
                for t in mod.get("topics", []):
                    topics.append(StudyTopic(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        key_points=t.get("key_points", []),
                        resources=t.get("resources", [])
                    ))
                modules.append(StudyModule(title=mod.get("title", ""), topics=topics))
        except Exception as e:
            print(f"Study plan parse error: {e}")

    if not modules:
        modules = [
            StudyModule(title="Core Fundamentals", topics=[
                StudyTopic(name="Foundation Concepts",
                           description=f"Master the core principles of {req.domain}.",
                           key_points=["Core terminology and concepts", "Industry standards and best practices",
                                       "Common tools and frameworks", "Hands-on practice projects"],
                           resources=["Book: Domain-specific textbook", "Course: Coursera/Udemy fundamentals"])
            ]),
            StudyModule(title="Practical Application", topics=[
                StudyTopic(name="Real-world Projects",
                           description="Apply your knowledge through practical projects.",
                           key_points=["Build a portfolio project", "Contribute to open source",
                                       "Code reviews and best practices", "Testing and documentation"],
                           resources=["GitHub: Explore open source projects", "Course: Project-based learning"])
            ]),
            StudyModule(title="Advanced Topics", topics=[
                StudyTopic(name="Advanced Techniques",
                           description="Deep dive into complex and specialized areas.",
                           key_points=["System design patterns", "Performance optimization",
                                       "Security considerations", "Scalability strategies"],
                           resources=["Book: Advanced system design", "Course: Architecture patterns"])
            ]),
            StudyModule(title="Interview Preparation", topics=[
                StudyTopic(name="Interview Readiness",
                           description="Prepare specifically for technical and behavioral interviews.",
                           key_points=["Common interview questions", "Problem-solving frameworks",
                                       "STAR method for behavioral questions", "Mock interview practice"],
                           resources=["LeetCode: Practice problems", "Pramp: Mock interviews"])
            ])
        ]

    return modules


# ═══════════════════════════════════════════════════════════════════════════════
#  MOCK INTERVIEW FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    question: str
    question_type: str   # "technical" | "behavioral" | "situational"
    user_answer: str
    role: str

class FeedbackResponse(BaseModel):
    score: int           # 0-100
    grade: str           # "Excellent" | "Good" | "Average" | "Needs Work"
    strengths: List[str]
    improvements: List[str]
    model_answer_snippet: str
    overall_feedback: str

@app.post("/api/mock/feedback", response_model=FeedbackResponse)
async def get_mock_feedback(req: FeedbackRequest):
    """Evaluate a mock interview answer using Gemini."""

    prompt = f"""
You are an expert hiring manager evaluating a candidate's interview answer.

Role being interviewed for: {req.role}
Question type: {req.question_type}
Question: {req.question}
Candidate's Answer: {req.user_answer}

Evaluate and return ONLY a JSON object with these fields:
- "score": integer 0-100 (be honest and fair)
- "grade": one of "Excellent", "Good", "Average", "Needs Work"
- "strengths": array of 2-3 specific strengths in their answer (strings)
- "improvements": array of 2-3 specific areas to improve (strings)
- "model_answer_snippet": a 2-3 sentence example of what an ideal answer would include (string)
- "overall_feedback": 2-3 sentence holistic assessment (string)

Scoring guide:
- 85-100: Excellent — well-structured, specific examples, quantified, clear communication
- 70-84: Good — solid content, minor gaps in specificity or structure
- 50-69: Average — relevant but lacks depth, specificity, or structure
- 0-49: Needs Work — off-topic, too brief, or missing key elements

Return RAW JSON only. No markdown, no code fences.
"""

    raw = ask_gemini(prompt, fallback="")

    if raw:
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            return FeedbackResponse(
                score=data.get("score", 60),
                grade=data.get("grade", "Average"),
                strengths=data.get("strengths", []),
                improvements=data.get("improvements", []),
                model_answer_snippet=data.get("model_answer_snippet", ""),
                overall_feedback=data.get("overall_feedback", "")
            )
        except Exception as e:
            print(f"Feedback parse error: {e}")

    # Fallback scoring
    words = len(req.user_answer.split())
    score = min(90, max(35, 45 + words))
    grade = "Excellent" if score >= 85 else "Good" if score >= 70 else "Average" if score >= 50 else "Needs Work"
    return FeedbackResponse(
        score=score,
        grade=grade,
        strengths=["You attempted to answer the question.", "Your response shows some relevant knowledge."],
        improvements=["Add specific examples and quantify your impact.", "Use the STAR method to structure your answer."],
        model_answer_snippet=f"An ideal answer would begin with a specific situation, describe the actions taken, and conclude with measurable results relevant to the {req.role} role.",
        overall_feedback="A solid start. Focus on adding specificity, quantified achievements, and clearer structure to make your answer more compelling to interviewers."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ROLE SUGGESTER  —  Auto-fills resume form fields from job title
# ═══════════════════════════════════════════════════════════════════════════════

class RoleSuggestRequest(BaseModel):
    role: str
    level: str = "mid"

class RoleSuggestResponse(BaseModel):
    skills: str           # comma-separated
    summary: str          # draft professional summary
    experience_template: str   # 3 bullet points as newline-separated string
    certifications: List[str]  # recommended certs
    salary_range: str     # e.g. "$90k–$130k"

@app.post("/api/suggest/role", response_model=RoleSuggestResponse)
async def suggest_role(req: RoleSuggestRequest):
    """Given a job title + level, return pre-filled suggestions for resume fields."""

    prompt = f"""
You are a career expert. For a {req.level}-level "{req.role}", provide resume form suggestions.

Return ONLY a JSON object with these exact fields:
- "skills": comma-separated list of 12-15 most important technical and soft skills (string)
- "summary": a 3-sentence professional summary draft (string, plain text, no markdown)
- "experience_template": 3 strong resume bullet points as a single string separated by newlines, starting with action verbs and including metrics (string)
- "certifications": array of 3-4 most valuable certifications to get (strings)
- "salary_range": typical salary range for this role and level in USD (string, e.g. "$80k-$110k")

No markdown, no code fences, return RAW JSON only.
"""

    raw = ask_gemini(prompt, fallback="")
    if raw:
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            return RoleSuggestResponse(
                skills=data.get("skills", ""),
                summary=data.get("summary", ""),
                experience_template=data.get("experience_template", ""),
                certifications=data.get("certifications", []),
                salary_range=data.get("salary_range", "")
            )
        except Exception as e:
            print(f"Role suggest parse error: {e}")

    # Fallback
    role_lower = req.role.lower()
    return RoleSuggestResponse(
        skills=f"Python, JavaScript, Problem Solving, Communication, Git, Agile, REST APIs, SQL, Docker, AWS, Team Leadership, System Design",
        summary=f"Results-driven {req.role} with strong expertise in industry-standard tools and frameworks. Proven track record of delivering high-quality solutions in collaborative, fast-paced environments. Passionate about continuous learning and driving measurable impact.",
        experience_template=f"Developed and maintained scalable {req.role} solutions, improving system performance by 30%.\nLed cross-functional collaboration with 5+ team members to deliver critical features on schedule.\nImplemented automated testing pipelines reducing production bugs by 40%.",
        certifications=["AWS Certified Solutions Architect", "Google Cloud Professional", "Certified Scrum Master"],
        salary_range="$75k–$120k"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  JOB DESCRIPTION PARSER  —  Extract skills & keywords from a JD
# ═══════════════════════════════════════════════════════════════════════════════

class JDParseRequest(BaseModel):
    job_description: str

class JDParseResponse(BaseModel):
    role: str
    required_skills: List[str]
    preferred_skills: List[str]
    keywords: List[str]
    summary: str
    match_tips: List[str]

@app.post("/api/jd/parse", response_model=JDParseResponse)
async def parse_job_description(req: JDParseRequest):
    """Parse a job description and extract key requirements for resume tailoring."""

    prompt = f"""
Analyze this job description and extract key information for resume tailoring:

JOB DESCRIPTION:
{req.job_description[:3000]}

Return ONLY a JSON object with:
- "role": detected job title (string)
- "required_skills": array of must-have technical skills mentioned (strings, max 10)
- "preferred_skills": array of nice-to-have skills (strings, max 8)
- "keywords": array of ATS keywords to include in resume (strings, max 12)
- "summary": 2-sentence summary of what the role needs (string)
- "match_tips": array of 3-4 specific tips to tailor a resume for this JD (strings)

Return RAW JSON only. No markdown, no code fences.
"""

    raw = ask_gemini(prompt, fallback="")
    if raw:
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            return JDParseResponse(
                role=data.get("role", "N/A"),
                required_skills=data.get("required_skills", []),
                preferred_skills=data.get("preferred_skills", []),
                keywords=data.get("keywords", []),
                summary=data.get("summary", ""),
                match_tips=data.get("match_tips", [])
            )
        except Exception as e:
            print(f"JD parse error: {e}")

    return JDParseResponse(
        role="Unknown Role",
        required_skills=["See job description for required skills"],
        preferred_skills=[],
        keywords=["leadership", "communication", "problem solving"],
        summary="Unable to parse the job description automatically. Please review it manually.",
        match_tips=["Tailor your skills section to match keywords in the JD.", "Quantify achievements with metrics.", "Mirror the language used in the job description."]
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  AI CHAT ASSISTANT  —  Conversational career help
# ═══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = ""   # e.g. current page, user's role

class ChatResponse(BaseModel):
    reply: str

@app.post("/api/chat", response_model=ChatResponse)
async def ai_chat(req: ChatRequest):
    """General AI career assistant chat."""

    system_ctx = f"Context: {req.context}" if req.context else ""

    prompt = f"""
You are Insignia AI, a friendly and expert career coach and interview preparation assistant.
{system_ctx}

User's message: {req.message}

Rules:
- Give concise, actionable, encouraging advice
- Focus on career, interviews, resumes, skills, and job search
- Use bullet points when listing items
- Max 250 words
- Be specific and practical, not generic
- If asked about a specific role, give role-specific advice

Respond in plain text (no markdown headers, minimal formatting).
"""

    reply = ask_gemini(prompt, fallback="I'm here to help with your career preparation! Ask me about resume writing, interview tips, skill development, or any career-related questions.")
    return ChatResponse(reply=reply)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status": "Insignia AI Backend is running!",
        "llm": "Google Gemini 2.0 Flash" if client else "Fallback mode (no API key)",
        "endpoints": [
            "POST /api/resume/generate",
            "POST /api/suggest/role",
            "POST /api/jd/parse",
            "POST /api/qa/generate",
            "POST /api/study/generate",
            "POST /api/mock/feedback",
            "POST /api/chat",
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok", "gemini_configured": bool(client)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
