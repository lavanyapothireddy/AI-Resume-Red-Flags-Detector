import os
import json
from groq import Groq
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file, filename):
    file_bytes = file.read()
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_bytes)
    elif ext == 'docx':
        return extract_text_from_docx(file_bytes)
    else:
        return file_bytes.decode('utf-8', errors='ignore')

SYSTEM_PROMPT = """You are an expert HR analyst and resume reviewer with 15+ years of experience. 
Your job is to analyze resumes and identify red flags, concerns, and areas of improvement.

Analyze the resume thoroughly and return a JSON response with this exact structure:
{
  "overall_score": <number 0-100, higher = fewer red flags>,
  "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "summary": "<2-3 sentence overall assessment>",
  "red_flags": [
    {
      "category": "<category name>",
      "severity": "<LOW|MEDIUM|HIGH|CRITICAL>",
      "title": "<short title>",
      "description": "<detailed explanation>",
      "suggestion": "<how to fix it>"
    }
  ],
  "strengths": ["<strength 1>", "<strength 2>", ...],
  "categories_analyzed": {
    "employment_gaps": "<PASS|FAIL|WARNING>",
    "job_hopping": "<PASS|FAIL|WARNING>",
    "skills_mismatch": "<PASS|FAIL|WARNING>",
    "education_issues": "<PASS|FAIL|WARNING>",
    "formatting_issues": "<PASS|FAIL|WARNING>",
    "ats_compatibility": "<PASS|FAIL|WARNING>",
    "exaggeration_claims": "<PASS|FAIL|WARNING>",
    "contact_info": "<PASS|FAIL|WARNING>"
  }
}

Categories to check:
- Employment Gaps: Unexplained gaps > 6 months
- Job Hopping: Multiple jobs < 1 year, especially recent ones
- Skills Mismatch: Listed skills don't match experience or job history
- Education Issues: Unverifiable degrees, unusual institutions, missing dates
- Formatting Issues: Poor structure, typos, inconsistency, hard to read
- ATS Compatibility: Use of tables, images, headers that break ATS parsing
- Exaggeration/Claims: Vague buzzwords, unquantified achievements, suspicious titles
- Contact Info: Missing email/phone, unprofessional email address

Return ONLY valid JSON, no markdown, no extra text."""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    resume_text = ""

    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PDF, DOCX, or TXT.'}), 400
        try:
            resume_text = extract_text(file, secure_filename(file.filename))
        except Exception as e:
            return jsonify({'error': f'Failed to parse file: {str(e)}'}), 400
    elif 'text' in request.form and request.form['text'].strip():
        resume_text = request.form['text'].strip()
    else:
        return jsonify({'error': 'Please provide a resume file or paste resume text.'}), 400

    if len(resume_text.strip()) < 100:
        return jsonify({'error': 'Resume text too short. Please provide a complete resume.'}), 400

    if len(resume_text) > 15000:
        resume_text = resume_text[:15000]

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=2000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this resume for red flags:\n\n{resume_text}"}
            ],
        )

        response_text = completion.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        result = json.loads(response_text)
        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({'error': 'Failed to parse AI response. Please try again.'}), 500
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)