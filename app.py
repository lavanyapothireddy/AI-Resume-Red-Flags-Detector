import os
import json
import logging
import traceback
import io
from groq import Groq
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    logger.warning("WARNING: GROQ_API_KEY is not set!")
client = Groq(api_key=GROQ_API_KEY)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_bytes):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

def extract_text_from_docx(file_bytes):
    import docx
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
Analyze the resume and return ONLY a JSON object with this exact structure, no markdown, no extra text:
{
  "overall_score": <number 0-100>,
  "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "summary": "<2-3 sentence assessment>",
  "red_flags": [
    {
      "category": "<category>",
      "severity": "<LOW|MEDIUM|HIGH|CRITICAL>",
      "title": "<short title>",
      "description": "<explanation>",
      "suggestion": "<how to fix>"
    }
  ],
  "strengths": ["<strength 1>", "<strength 2>"],
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
}"""

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resume Red Flag Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0a0f;--surface:#111118;--surface2:#18181f;--border:#2a2a35;--accent:#ff3c3c;--accent2:#ff7b00;--green:#00e599;--text:#f0f0f5;--muted:#6b6b80}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh}
.container{max-width:960px;margin:0 auto;padding:0 24px}
header{padding:48px 0 40px;border-bottom:1px solid var(--border)}
.logo-row{display:flex;align-items:center;gap:16px;margin-bottom:20px}
.logo-icon{width:48px;height:48px;background:var(--accent);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:24px;animation:glow 2s ease-in-out infinite}
@keyframes glow{0%,100%{box-shadow:0 0 30px rgba(255,60,60,0.4)}50%{box-shadow:0 0 60px rgba(255,60,60,0.7)}}
.logo-text{font-family:'Syne',sans-serif;font-size:22px;font-weight:800}
.logo-text span{color:var(--accent)}
header h1{font-family:'Syne',sans-serif;font-size:clamp(30px,6vw,52px);font-weight:800;line-height:1.1;letter-spacing:-2px;margin-bottom:12px}
header h1 em{font-style:normal;color:var(--accent)}
header p{color:var(--muted);font-size:16px;max-width:520px;line-height:1.6}
.upload-section{padding:48px 0}
.tabs{display:flex;border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-bottom:28px;width:fit-content}
.tab{padding:10px 24px;font-family:'DM Mono',monospace;font-size:13px;cursor:pointer;border:none;background:transparent;color:var(--muted);transition:all .2s}
.tab.active{background:var(--accent);color:#fff}
.tab:hover:not(.active){background:var(--surface2);color:var(--text)}
.tab-content{display:none}
.tab-content.active{display:block}
.dropzone{border:2px dashed var(--border);border-radius:16px;padding:60px 40px;text-align:center;cursor:pointer;transition:all .3s;background:var(--surface)}
.dropzone:hover,.dropzone.dragover{border-color:var(--accent);background:var(--surface2);box-shadow:0 0 40px rgba(255,60,60,.1)}
.dropzone-icon{font-size:48px;margin-bottom:16px}
.dropzone h3{font-family:'Syne',sans-serif;font-size:20px;font-weight:700;margin-bottom:8px}
.dropzone p{color:var(--muted);font-size:14px}
.badges{margin-top:16px;display:flex;gap:8px;justify-content:center}
.badge{font-family:'DM Mono',monospace;font-size:11px;padding:4px 10px;border:1px solid var(--border);border-radius:6px;color:var(--muted)}
#file-input{display:none}
textarea{width:100%;min-height:280px;background:var(--surface);border:1px solid var(--border);border-radius:16px;color:var(--text);font-family:'DM Mono',monospace;font-size:13px;line-height:1.7;padding:24px;resize:vertical;outline:none;transition:border-color .2s}
textarea:focus{border-color:var(--accent)}
textarea::placeholder{color:var(--muted)}
.file-sel{background:var(--surface2);border:1px solid var(--green);border-radius:12px;padding:16px 20px;display:flex;align-items:center;gap:12px;margin-top:16px}
.file-sel .fname{font-family:'DM Mono',monospace;font-size:13px}
.file-sel .rm{margin-left:auto;cursor:pointer;color:var(--muted);font-size:18px}
.file-sel .rm:hover{color:var(--accent)}
.analyze-btn{margin-top:24px;width:100%;padding:18px;background:var(--accent);color:#fff;border:none;border-radius:12px;font-family:'Syne',sans-serif;font-size:18px;font-weight:700;cursor:pointer;transition:all .2s}
.analyze-btn:hover{background:#e62e2e;transform:translateY(-1px);box-shadow:0 8px 30px rgba(255,60,60,.4)}
.analyze-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.loading{display:none;text-align:center;padding:80px 0}
.loading.visible{display:block}
.scanner{width:80px;height:80px;margin:0 auto 24px;position:relative}
.sr{width:80px;height:80px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin 1s linear infinite;position:absolute;inset:0}
.sr:nth-child(2){width:60px;height:60px;top:10px;left:10px;border-top-color:var(--accent2);animation-duration:.7s;animation-direction:reverse}
@keyframes spin{to{transform:rotate(360deg)}}
.loading h3{font-family:'Syne',sans-serif;font-size:20px;font-weight:700;margin-bottom:8px}
.loading p{color:var(--muted);font-size:14px}
.results{display:none;padding-bottom:80px}
.results.visible{display:block}
.score-card{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:40px;margin-bottom:28px;display:grid;grid-template-columns:auto 1fr auto;gap:40px;align-items:center}
.score-circle{width:120px;height:120px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;border:3px solid}
.score-num{font-family:'Syne',sans-serif;font-size:40px;font-weight:800;line-height:1}
.score-lbl{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace}
.score-info h2{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;margin-bottom:10px}
.score-info p{color:var(--muted);line-height:1.6;font-size:15px}
.risk-badge{font-family:'Syne',sans-serif;font-weight:800;font-size:14px;padding:8px 20px;border-radius:100px;letter-spacing:2px;text-transform:uppercase}
.sec-title{font-family:'Syne',sans-serif;font-size:20px;font-weight:700;margin-bottom:20px;display:flex;align-items:center;gap:10px}
.sec-title::after{content:'';flex:1;height:1px;background:var(--border)}
.cat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:32px}
.cat-item{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;display:flex;align-items:center;gap:12px}
.cat-name{font-size:13px;font-weight:500}
.cat-lbl{font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);margin-top:2px}
.flags-list{margin-bottom:32px;display:flex;flex-direction:column;gap:16px}
.flag-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;overflow:hidden}
.flag-hdr{padding:20px 24px;display:flex;align-items:center;gap:16px;cursor:pointer}
.sdot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.fcat{font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}
.ftitle{font-family:'Syne',sans-serif;font-size:16px;font-weight:700}
.stag{margin-left:auto;font-family:'DM Mono',monospace;font-size:11px;padding:4px 12px;border-radius:6px}
.flag-body{padding:16px 24px 20px;border-top:1px solid var(--border);display:none}
.flag-body.open{display:block}
.fdesc{color:var(--muted);font-size:14px;line-height:1.7;margin-bottom:14px}
.sug{background:var(--surface2);border-left:3px solid var(--green);border-radius:0 8px 8px 0;padding:12px 16px;font-size:14px;line-height:1.6}
.sug strong{color:var(--green);font-family:'DM Mono',monospace;font-size:11px;letter-spacing:1px;display:block;margin-bottom:4px}
.str-grid{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:32px}
.str-tag{background:rgba(0,229,153,.1);border:1px solid rgba(0,229,153,.3);color:var(--green);padding:8px 16px;border-radius:100px;font-size:14px}
.reanalyze{display:inline-flex;align-items:center;gap:8px;padding:12px 24px;background:var(--surface2);border:1px solid var(--border);color:var(--text);border-radius:10px;font-family:'Syne',sans-serif;font-size:15px;font-weight:700;cursor:pointer;transition:all .2s;margin-bottom:32px}
.reanalyze:hover{border-color:var(--accent);color:var(--accent)}
.err-box{display:none;background:rgba(255,45,45,.1);border:1px solid rgba(255,45,45,.4);border-radius:12px;padding:20px;margin-top:16px;color:#ff6b6b;font-size:14px;line-height:1.6}
.err-box.visible{display:block}
@media(max-width:640px){.score-card{grid-template-columns:1fr;text-align:center}.risk-badge{width:fit-content;margin:0 auto}}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo-row"><div class="logo-icon">🚩</div><div class="logo-text">Resume<span>Scan</span></div></div>
    <h1>AI <em>Red Flag</em><br>Detector</h1>
    <p>Upload any resume for instant AI analysis — red flags, ATS compatibility, and actionable suggestions.</p>
  </header>

  <div class="upload-section" id="upload-section">
    <div class="tabs">
      <button class="tab active" onclick="switchTab('file')">📁 Upload File</button>
      <button class="tab" onclick="switchTab('paste')">📋 Paste Text</button>
    </div>
    <div class="tab-content active" id="tab-file">
      <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
        <div class="dropzone-icon">📄</div>
        <h3>Drop your resume here</h3>
        <p>or click to browse</p>
        <div class="badges">
          <span class="badge">PDF</span><span class="badge">DOCX</span><span class="badge">TXT</span>
        </div>
      </div>
      <input type="file" id="file-input" accept=".pdf,.docx,.txt">
      <div class="file-sel" id="file-sel" style="display:none">
        <span>✅</span><span class="fname" id="fname"></span>
        <span class="rm" onclick="removeFile()">✕</span>
      </div>
    </div>
    <div class="tab-content" id="tab-paste">
      <textarea id="resume-text" placeholder="Paste your full resume here..."></textarea>
    </div>
    <div class="err-box" id="err-box"></div>
    <button class="analyze-btn" onclick="analyzeResume()">🔍 Analyze for Red Flags</button>
  </div>

  <div class="loading" id="loading">
    <div class="scanner"><div class="sr"></div><div class="sr"></div></div>
    <h3>Scanning your resume...</h3>
    <p>Analyzing 8 categories of red flags with AI</p>
  </div>

  <div class="results" id="results">
    <button class="reanalyze" onclick="resetApp()">← Analyze Another</button>
    <div class="score-card">
      <div class="score-circle" id="score-circle"><div class="score-num" id="score-num">--</div><div class="score-lbl">SCORE</div></div>
      <div class="score-info"><h2 id="r-heading">Analysis Complete</h2><p id="r-summary"></p></div>
      <div class="risk-badge" id="risk-badge">--</div>
    </div>
    <div class="sec-title">Category Breakdown</div>
    <div class="cat-grid" id="cat-grid"></div>
    <div class="sec-title" id="flags-title">Red Flags</div>
    <div class="flags-list" id="flags-list"></div>
    <div class="sec-title" id="str-title">Strengths</div>
    <div class="str-grid" id="str-grid"></div>
  </div>
</div>
<script>
let selFile=null,activeTab='file';
function switchTab(t){
  activeTab=t;
  document.querySelectorAll('.tab').forEach((el,i)=>el.classList.toggle('active',(i===0&&t==='file')||(i===1&&t==='paste')));
  document.getElementById('tab-file').classList.toggle('active',t==='file');
  document.getElementById('tab-paste').classList.toggle('active',t==='paste');
}
const dz=document.getElementById('dropzone');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('dragover')});
dz.addEventListener('dragleave',()=>dz.classList.remove('dragover'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('dragover');if(e.dataTransfer.files[0])setFile(e.dataTransfer.files[0])});
document.getElementById('file-input').addEventListener('change',e=>{if(e.target.files[0])setFile(e.target.files[0])});
function setFile(f){
  selFile=f;
  document.getElementById('fname').textContent=f.name+' ('+(f.size/1024).toFixed(1)+' KB)';
  document.getElementById('file-sel').style.display='flex';
  dz.style.display='none';
}
function removeFile(){
  selFile=null;document.getElementById('file-input').value='';
  document.getElementById('file-sel').style.display='none';dz.style.display='block';
}
async function analyzeResume(){
  document.getElementById('err-box').classList.remove('visible');
  const fd=new FormData();
  if(activeTab==='file'){if(!selFile){showErr('Please select a file.');return;}fd.append('file',selFile);}
  else{const t=document.getElementById('resume-text').value.trim();if(!t){showErr('Please paste your resume.');return;}fd.append('text',t);}
  document.getElementById('upload-section').style.display='none';
  document.getElementById('loading').classList.add('visible');
  document.getElementById('results').classList.remove('visible');
  try{
    const res=await fetch('/analyze',{method:'POST',body:fd});
    const data=await res.json();
    if(!res.ok||data.error)throw new Error(data.error||'Analysis failed');
    render(data);
  }catch(e){
    document.getElementById('upload-section').style.display='block';
    document.getElementById('loading').classList.remove('visible');
    showErr(e.message);
  }
}
const FC={LOW:'#00e599',MEDIUM:'#ffdd00',HIGH:'#ff7b00',CRITICAL:'#ff2d2d'};
const SI={PASS:'✅',FAIL:'❌',WARNING:'⚠️'};
const CL={employment_gaps:'Employment Gaps',job_hopping:'Job Hopping',skills_mismatch:'Skills Match',education_issues:'Education',formatting_issues:'Formatting',ats_compatibility:'ATS Friendly',exaggeration_claims:'Authenticity',contact_info:'Contact Info'};
const HM={LOW:'✅ Looks Good!',MEDIUM:'⚠️ Some Concerns',HIGH:'🚨 Major Red Flags',CRITICAL:'🔴 Critical Issues'};
function render(d){
  document.getElementById('loading').classList.remove('visible');
  const score=d.overall_score||0,risk=d.risk_level||'UNKNOWN',col=FC[risk]||'#6b6b80';
  const sc=document.getElementById('score-circle');
  sc.style.borderColor=col;
  document.getElementById('score-num').style.color=col;
  document.getElementById('score-num').textContent=score;
  document.getElementById('r-heading').textContent=HM[risk]||'Analysis Complete';
  document.getElementById('r-summary').textContent=d.summary||'';
  const rb=document.getElementById('risk-badge');
  rb.textContent=risk+' RISK';rb.style.cssText=`background:${col}22;color:${col};border:1px solid ${col}44`;
  const cg=document.getElementById('cat-grid');cg.innerHTML='';
  for(const[k,v]of Object.entries(d.categories_analyzed||{}))
    cg.innerHTML+=`<div class="cat-item"><div style="font-size:18px">${SI[v]||'❓'}</div><div><div class="cat-name">${CL[k]||k}</div><div class="cat-lbl">${v}</div></div></div>`;
  const flags=d.red_flags||[];
  document.getElementById('flags-title').textContent=`🚩 Red Flags Found (${flags.length})`;
  const fl=document.getElementById('flags-list');fl.innerHTML='';
  if(!flags.length){fl.innerHTML='<div style="color:var(--green);padding:20px 0">✅ No significant red flags!</div>';}
  else flags.forEach((f,i)=>{
    const c=FC[f.severity]||'#6b6b80';
    fl.innerHTML+=`<div class="flag-card"><div class="flag-hdr" onclick="tf(${i})"><div class="sdot" style="background:${c}"></div><div><div class="fcat">${f.category||''}</div><div class="ftitle">${f.title||''}</div></div><div class="stag" style="background:${c}22;color:${c};border:1px solid ${c}44">${f.severity}</div></div><div class="flag-body" id="fb${i}"><div class="fdesc">${f.description||''}</div>${f.suggestion?`<div class="sug"><strong>💡 SUGGESTION</strong>${f.suggestion}</div>`:''}</div></div>`;
  });
  const str=d.strengths||[];
  document.getElementById('str-title').textContent=`💪 Strengths (${str.length})`;
  document.getElementById('str-grid').innerHTML=str.map(s=>`<div class="str-tag">✓ ${s}</div>`).join('');
  document.getElementById('results').classList.add('visible');
}
function tf(i){document.getElementById('fb'+i).classList.toggle('open')}
function resetApp(){
  document.getElementById('results').classList.remove('visible');
  document.getElementById('upload-section').style.display='block';
  removeFile();document.getElementById('resume-text').value='';
}
function showErr(m){const b=document.getElementById('err-box');b.textContent='⚠️ '+m;b.classList.add('visible')}
</script>
</body>
</html>"""

@app.route('/')
def index():
    return HTML_PAGE, 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route('/health')
def health():
    return jsonify({"status": "ok", "groq_key_set": bool(GROQ_API_KEY)})

@app.route('/analyze', methods=['POST'])
def analyze():
    response_text = ""
    try:
        resume_text = ""

        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Use PDF, DOCX, or TXT.'}), 400
            try:
                resume_text = extract_text(file, secure_filename(file.filename))
            except Exception as e:
                logger.error(f"File extraction error: {traceback.format_exc()}")
                return jsonify({'error': f'Failed to read file: {str(e)}'}), 400
        elif 'text' in request.form and request.form['text'].strip():
            resume_text = request.form['text'].strip()
        else:
            return jsonify({'error': 'Please provide a resume file or paste text.'}), 400

        if len(resume_text.strip()) < 50:
            return jsonify({'error': 'Resume text too short. Please provide a complete resume.'}), 400

        if len(resume_text) > 12000:
            resume_text = resume_text[:12000]

        logger.info(f"Sending to Groq, text length={len(resume_text)}")

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=2000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this resume:\n\n{resume_text}"}
            ],
            temperature=0.2,
        )

        response_text = completion.choices[0].message.content.strip()
        logger.info(f"Got Groq response, length={len(response_text)}")

        # Strip markdown fences if present
        if "```" in response_text:
            for part in response_text.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    response_text = part
                    break

        result = json.loads(response_text)
        return jsonify(result)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}. Raw: {response_text[:300]}")
        return jsonify({'error': 'AI returned malformed response. Please try again.'}), 500
    except Exception as e:
        logger.error(f"Analyze error: {traceback.format_exc()}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
