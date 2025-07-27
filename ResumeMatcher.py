import requests
import re
import spacy
import importlib
import ner_keywords_lists
from ner_keywords_lists import it_skills
from ner_keywords_lists import degree_list
from ner_keywords_lists import task_keywords
from ner_keywords_lists import departments
from ner_keywords_lists import experience_keywords
import fitz
from numpy import dot
from numpy.linalg import norm
from dotenv import load_dotenv
import os
import numpy as np


load_dotenv()
hf_token = os.getenv("HF_API_TOKEN")
nlp=spacy.load("en_core_web_sm")

def mean_pooling(embedding):
    if isinstance(embedding,list) and len(embedding)>1 and isinstance(embedding[0],list):
        return np.mean(embedding, axis=0).tolist()
    return embedding

def extract_sentences(text1,entities):
    doc = nlp(text1)
    sentences = []
    for sent in doc.sents:
        for token in sent:
            if token.text.lower() in entities:
                sentences.append(sent.text)
                break
    return sentences

def extract_sentences_degree(text1, entities):
    doc = nlp(text1)
    sentences = []
    for sent in doc.sents:
        sent_text_lower = sent.text.lower()
        for entity in entities:
            if entity.lower() in sent_text_lower:
                sentences.append(entity.lower())
    return sentences

def extract_sentences_substrings(text1,entities):
    doc = nlp(text1)
    sentences = []
    for sent in doc.sents:
        sent_text=sent.text.lower()
        if any(entity in sent_text for entity in entities):
            sentences.append(sent_text)
    return sentences

def get_embedding(text1,text2):
    API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/sentence-similarity"
    headers = {
        "Authorization": f"Bearer {hf_token}"
    }
    payload = {
    "inputs": {
        "source_sentence": text2,
        "sentences": text1
    }
    }
    try:
        response = requests.post(API_URL,headers=headers,json=payload)
        return response.json()
    except Exception as e:
        print("Failed to parse JSON:", e)


def get_cosine_similarity(embd1,embd2):
    return dot(embd1,embd2)/(norm(embd1)*norm(embd2))

def ResumeMatcher(ResumeURLs,JD):
    resume_skill_sents=[]
    job_skill_sents=[]
    resume_degree_sents=[]
    job_degree_sents=[]
    resume_project_sents=[]
    job_project_sents=[]
    resume_experience_sents=[]
    job_experience_sents=[]
    for ResumeURL in ResumeURLs:
        response = requests.get(ResumeURL)
        with open("./resume_temp.pdf","wb") as f:
            f.write(response.content)
        doc = fitz.open("./resume_temp.pdf")
        resume_text = ""
        for page in doc:
            resume_text+=page.get_text()
        job_desc = JD
        resume_skill = extract_sentences(resume_text,it_skills)
        resume_degree = extract_sentences_degree(resume_text,degree_list) + extract_sentences_degree(resume_text,departments)
        resume_project = extract_sentences_substrings(resume_text,task_keywords)
        resume_experience = extract_sentences_substrings(resume_text,experience_keywords)
        resume_skill_sents.append(" ".join(resume_skill))
        resume_degree_sents.append(" ".join(resume_degree))
        resume_project_sents.append(" ".join(resume_project))
        resume_experience_sents.append(" ".join(resume_experience))
    job_skill = extract_sentences(job_desc,it_skills)
    job_degree = extract_sentences_degree(job_desc,degree_list) + extract_sentences_degree(job_desc,departments)
    job_project = extract_sentences_substrings(job_desc,task_keywords)
    job_experience = extract_sentences_substrings(job_desc,experience_keywords)
    job_skill_sents = (" ".join(job_skill))
    job_degree_sents = (" ".join(job_degree))
    job_project_sents = (" ".join(job_project))
    job_experience_sents = (" ".join(job_experience))
    skill_score = get_embedding(resume_skill_sents,job_skill_sents)
    degree_score = get_embedding(resume_degree_sents,job_degree_sents)
    experience_score = get_embedding(resume_project_sents,job_project_sents)
    project_score = get_embedding(resume_experience_sents,job_experience_sents)
    result = []
    for i in range(0,len(skill_score)):
        skill_wt = 0.5
        degree_wt = 0.1
        exp_wt = 0.2
        pro_wt = 0.2
        if skill_score[i] >= 0.7:
            skill_wt=0.6
            exp_wt=0.15
            pro_wt=0.15
        total_score = (
        skill_wt * skill_score[i] +
        degree_wt * degree_score[i] +
        exp_wt * experience_score[i] +
        pro_wt * project_score[i]
        )
        result.append(round(total_score*100))
    return {"result":result,"skill_score":skill_score,"degree_score":degree_score,"experience_score":experience_score,"project_score":project_score}

    