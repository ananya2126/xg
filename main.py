from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

# Add src folder to path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../src"
        )
    )
)

# Import services
from parsing import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt
)

from skills import extract_skills

from ner_skill_extractor import extract_skills_ner

from llm_enhancer import enhance_resume_section

from learning_resources import get_learning_resources

from project_ideas import generate_project_ideas

from fit_classifier import predict_fit


# FastAPI instance
app = FastAPI(
    title="Smart Career Advisor API"
)


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------
# Utility function to extract text
# --------------------------------
def extract_text(file: UploadFile):

    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file.file)

    elif filename.endswith(".docx"):
        return extract_text_from_docx(file.file)

    elif filename.endswith(".txt"):
        return extract_text_from_txt(file.file)

    else:
        return ""


# --------------------------------
# Health check
# --------------------------------
@app.get("/")
def health():

    return {
        "status": "API running"
    }


# --------------------------------
# Extract text
# --------------------------------
@app.post("/extract-text")
async def extract_text_api(
    file: UploadFile = File(...)
):

    text = extract_text(file)

    return {
        "filename": file.filename,
        "text": text
    }


# --------------------------------
# Extract skills
# --------------------------------
class TextInput(BaseModel):

    text: str


@app.post("/extract-skills")
def extract_skills_api(data: TextInput):

    try:

        skills = extract_skills_ner(data.text)

    except:

        skills = extract_skills(data.text)

    return {

        "skills": skills,

        "count": len(skills)

    }


# --------------------------------
# Skill match
# --------------------------------
class MatchRequest(BaseModel):

    resume_text: str

    job_description: str


@app.post("/skill-match")
def skill_match(data: MatchRequest):

    resume_skills = extract_skills(data.resume_text)

    jd_skills = extract_skills(data.job_description)

    matched = list(
        set(resume_skills)
        &
        set(jd_skills)
    )

    missing = list(
        set(jd_skills)
        -
        set(resume_skills)
    )

    score = (
        len(matched) /
        len(jd_skills) * 100
        if jd_skills else 0
    )

    return {

        "match_score": score,

        "matched_skills": matched,

        "missing_skills": missing

    }


# --------------------------------
# ML prediction from TEXT
# --------------------------------
@app.post("/predict-fit")
def predict_fit_api(data: MatchRequest):

    result = predict_fit(

        resume_text=data.resume_text,

        job_description=data.job_description

    )

    return result


# --------------------------------
# ML prediction from FILES (IMPORTANT)
# --------------------------------
@app.post("/predict-fit-file")
async def predict_fit_file(

    resume: UploadFile = File(...),

    job_description: UploadFile = File(...)

):

    resume_text = extract_text(resume)

    job_text = extract_text(job_description)

    result = predict_fit(

        resume_text=resume_text,

        job_description=job_text

    )

    return result


# --------------------------------
# Resume enhancement
# --------------------------------
class EnhanceRequest(BaseModel):

    resume_text: str

    job_description: str

    missing_skills: list


@app.post("/enhance-resume")
def enhance_resume_api(data: EnhanceRequest):

    improved = enhance_resume_section(

        data.resume_text,

        data.job_description,

        data.missing_skills

    )

    return {

        "enhanced_resume": improved

    }


# --------------------------------
# Learning resources
# --------------------------------
class SkillsRequest(BaseModel):

    skills: list


@app.post("/learning-resources")
def learning_resources_api(data: SkillsRequest):

    resources = get_learning_resources(data.skills)

    return resources


# --------------------------------
# Project ideas
# --------------------------------
class ProjectRequest(BaseModel):

    resume_text: str

    skills: list


@app.post("/project-ideas")
def project_ideas_api(data: ProjectRequest):

    ideas = generate_project_ideas(

        data.resume_text,

        data.skills

    )

    return {

        "project_ideas": ideas

    }
@app.post("/full-analysis")
async def full_analysis(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...)
):

    resume_text = extract_text(resume)
    job_text = extract_text(job_description)
    # Extract skills
    try:
        resume_skills = extract_skills_ner(resume_text)
    except:
        resume_skills = extract_skills(resume_text)

    try:
        jd_skills = extract_skills_ner(job_text)
    except:
        jd_skills = extract_skills(job_text)

    # Skill matching
    matched_skills = list(
        set(resume_skills) & set(jd_skills)
    )

    missing_skills = list(
        set(jd_skills) - set(resume_skills)
    )

    match_score = (
        len(matched_skills) /
        len(jd_skills) * 100
        if jd_skills else 0
    )

    # ML prediction
    prediction_result = predict_fit(
        resume_text=resume_text,
        job_description=job_text
    )

    # Learning resources
    resources = get_learning_resources(
        missing_skills
    )

    # Project ideas
    project_ideas = generate_project_ideas(
        resume_text[:1500],   # 🔥 reduce more
        resume_skills[:20] 
    )

    return {

        "prediction": prediction_result,

        "match_score": match_score,

        "resume_skills": resume_skills,

        "job_skills": jd_skills,

        "matched_skills": matched_skills,

        "missing_skills": missing_skills,

        "learning_resources": resources,

        "project_ideas": project_ideas

    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
