from flask import Flask
from ResumeMatcher import ResumeMatcher
import requests
from flask import jsonify,request

app = Flask(__name__)

@app.route('/')
def index():
    return "hello"

@app.post('/get_score')
def get_score():
    data = request.get_json()
    resume_url = data.get("resume_url")
    job_desc = data.get("job_description")
    if not resume_url or not job_desc or len(resume_url)==0:
        return jsonify({'error':'All fields are required'}),400
    score = ResumeMatcher(resume_url,job_desc)
    return jsonify({'score':score})

if __name__ == "__main__":
    app.run(debug=True)