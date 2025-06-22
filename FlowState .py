# flowstate_ai.py - Complete Freestyle Analysis System
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import openai
import re
import uuid
import datetime
import os
import sqlite3
import json
import tempfile
import speech_recognition as sr
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from pydub import AudioSegment
from io import BytesIO

# Configuration
load_dotenv()
app = Flask(__name__)
limiter = Limiter(app=app, key_func=get_remote_address)

# Security checks
if not os.getenv('OPENAI_API_KEY'):
    print("⚠️ Running in LOCAL-ONLY mode (no AI)")
    openai.api_key = "mock_key_for_dev"
else:
    openai.api_key = os.getenv('OPENAI_API_KEY')

# Constants
OPENAI_MODEL = "gpt-3.5-turbo-0125"
MAX_TOKENS = 200
POWER_WORDS = ["truth", "pain", "legacy", "dream", "freedom", "grind", "mirror", "soul", "real", "life", "mind"]

class FlowStateEngine:
    def __init__(self):
        self.reset_session()
        self.init_db()
    
    def reset_session(self):
        self.session_id = str(uuid.uuid4())
        self.timestamp = datetime.datetime.now()
        self.freestyle_history = []
        self.hook_bank = defaultdict(list)
        self.dejavu_lines = set()
        self.key_quotes = []
        self.flow_type = None
        self.current_bpm = None
    
    def init_db(self):
        self.conn = sqlite3.connect('flowstate.db', check_same_thread=False)
        self.c = self.conn.cursor()
        self.c.execute('''CREATE TABLE IF NOT EXISTS sessions
                         (id TEXT PRIMARY KEY,
                          data TEXT,
                          timestamp TEXT,
                          artist TEXT)''')
        self.conn.commit()
    
    def save_session(self, artist_name=None):
        session_data = {
            'history': self.freestyle_history,
            'hooks': self.hook_bank,
            'quotes': self.key_quotes,
            'flow_type': self.flow_type,
            'bpm': self.current_bpm
        }
        self.c.execute('''INSERT INTO sessions VALUES (?, ?, ?, ?)''',
                      (self.session_id, json.dumps(session_data), 
                       str(self.timestamp), artist_name))
        self.conn.commit()
    
    def load_session(self, session_id):
        self.c.execute('''SELECT data FROM sessions WHERE id=?''', (session_id,))
        data = self.c.fetchone()
        if data:
            session = json.loads(data[0])
            self.freestyle_history = session['history']
            self.hook_bank = defaultdict(list, session['hooks'])
            self.key_quotes = session['quotes']
            self.flow_type = session['flow_type']
            self.current_bpm = session['bpm']
            return True
        return False
    
    def analyze_with_openai(self, text):
        """Cost-optimized AI analysis with flow typing"""
        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": """Analyze rap lyrics. Return:
                     - Hook score (1-5)
                     - Flow type (Chopper/Melodic/Storytelling)
                     - Top 3 power words
                     - Concise improvement tips"""},
                    {"role": "user", "content": text[:500]}
                ],
                temperature=0.7,
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return None
    
    def process_line(self, line):
        line = line.strip()
        if not line: return None
            
        result = {
            "line": line,
            "deja_vu": False,
            "hook": False,
            "key_quote": False,
            "analysis": None,
            "flow_score": self._calc_flow_score(line),
            "power_words": [w for w in POWER_WORDS if w in line.lower()]
        }
        
        # Store history
        self.freestyle_history.append(line)
        
        # Rule-based analysis
        normalized_line = re.sub(r'[^a-zA-Z0-9]', '', line.lower())
        result["deja_vu"] = normalized_line in self.dejavu_lines
        self.dejavu_lines.add(normalized_line)
        
        # Hook detection
        if (len(line.split()) <= 8 and (
            any(t in line.lower() for t in ["that's", "you know", "check it"]) or
            line.endswith("?") or len(re.findall(r'\b\w+\b.*\b\1\b', line)) > 0)):
            result["hook"] = True
            self.hook_bank['hook'].append(line)
            
        # Key quotes
        if len(line.split()) > 6 and any(w in line.lower() for w in POWER_WORDS):
            result["key_quote"] = True
            self.key_quotes.append(line)
        
        # Only use AI every 3 lines to save costs
        if len(self.freestyle_history) % 3 == 0:
            analysis = self.analyze_with_openai("\n".join(self.freestyle_history[-3:]))
            result["analysis"] = analysis
            # Update flow type from analysis
            if "Chopper" in analysis: self.flow_type = "Chopper"
            elif "Melodic" in analysis: self.flow_type = "Melodic"
            elif "Storytelling" in analysis: self.flow_type = "Storytelling"
            
        return result
    
    def _calc_flow_score(self, line):
        syllables = sum(len(re.findall(r'[aeiouy]+', w)) for w in line.split())
        return min(10, max(1, int(syllables / max(1, len(line.split())/3))))
    
    def generate_song_structure(self):
        """Generate structure with minimal API calls"""
        if len(self.hook_bank['hook']) >= 2:
            return self._rule_based_structure()
        
        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Create concise song structure with:" + 
                     "1. Intro\n2. Verses\n3. Hooks\n4. Outro\nMax 150 tokens"},
                    {"role": "user", "content": "\n".join(self.freestyle_history[-12:])[:800]}
                ],
                temperature=0.7,
                max_tokens=150
            )
            structure = response.choices[0].message.content
            self.current_bpm = self._estimate_bpm()
            return {
                "structure": structure,
                "bpm": self.current_bpm,
                "flow_type": self.flow_type
            }
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return self._rule_based_structure()
    
    def _rule_based_structure(self):
        hooks = self.hook_bank['hook']
        verses = [l for l in self.freestyle_history if l not in hooks]
        self.current_bpm = self._estimate_bpm()
        return {
            "structure": (
                f"[INTRO]\n{verses[0] if verses else '...'}\n\n" +
                f"[VERSE 1]\n{'\n'.join(verses[1:4]) if len(verses) >=4 else '...'}\n\n" +
                f"[HOOK]\n{hooks[0] if hooks else '...'}\n\n" +
                f"[VERSE 2]\n{'\n'.join(verses[4:8]) if len(verses) >=8 else '...'}\n\n" +
                f"[OUTRO]\n{self.key_quotes[0] if self.key_quotes else verses[-1] if verses else '...'}"
            ),
            "bpm": self.current_bpm,
            "flow_type": self.flow_type
        }
    
    def _estimate_bpm(self):
        avg = np.mean([len(l.split()) for l in self.freestyle_history[-4:]]) if self.freestyle_history else 8
        return f"{int(80 + avg*5)}-{int(90 + avg*5)}"
    
    def find_matching_beat(self):
        """Locate matching BPM beat file"""
        if not self.current_bpm: return None
        target_bpm = int(self.current_bpm.split('-')[0])
        for f in os.listdir('static/beats'):
            if f.startswith(f"bpm_{target_bpm}"):
                return f"/static/beats/{f}"
        return None

engine = FlowStateEngine()

# API Endpoints
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@limiter.limit("5/minute")
def analyze():
    data = request.json
    line = data.get('line', '').strip()
    result = engine.process_line(line)
    return jsonify(result)

@app.route('/reset', methods=['POST'])
def reset():
    engine.reset_session()
    return jsonify({"status": "success", "session_id": engine.session_id})

@app.route('/save', methods=['POST'])
def save():
    artist = request.json.get('artist')
    engine.save_session(artist)
    return jsonify({"status": "saved", "session_id": engine.session_id})

@app.route('/load/<session_id>', methods=['GET'])
def load(session_id):
    if engine.load_session(session_id):
        return jsonify({"status": "loaded"})
    return jsonify({"error": "Session not found"}), 404

@app.route('/generate', methods=['GET'])
def generate():
    return jsonify(engine.generate_song_structure())

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio"}), 400
    
    audio_file = request.files['audio']
    temp_path = tempfile.mktemp(suffix='.wav')
    audio_file.save(temp_path)
    
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            
            # Create audio preview
            audio_blob = AudioSegment.from_wav(temp_path)
            buffer = BytesIO()
            audio_blob.export(buffer, format="mp3")
            buffer.seek(0)
            
            return jsonify({
                "text": text,
                "audio_preview": buffer.read().hex()  # Simple way to send binary data
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)

@app.route('/beats/<path:filename>')
def serve_beat(filename):
    return send_from_directory('static/beats', filename)

# HTML Template
@app.route('/template')
def template():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FlowState AI</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6a11cb;
                --secondary: #2575fc;
                --dark: #121212;
                --card: #1e1e1e;
                --text: #e0e0e0;
            }
            body {
                font-family: 'Poppins', sans-serif;
                background: var(--dark);
                color: var(--text);
                padding: 20px;
                line-height: 1.6;
            }
            .container {
                max-width: 100%;
                margin: 0 auto;
            }
            h1 {
                background: linear-gradient(to right, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                text-align: center;
                margin-bottom: 10px;
            }
            .card {
                background: var(--card);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            textarea {
                width: 100%;
                background: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 8px;
                padding: 12px;
                color: white;
                margin-bottom: 10px;
                min-height: 100px;
            }
            .btn {
                background: linear-gradient(to right, var(--primary), var(--secondary));
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 8px;
                margin-right: 8px;
                margin-bottom: 8px;
                font-weight: 600;
                cursor: pointer;
            }
            .tag {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 12px;
                margin-right: 5px;
                margin-bottom: 5px;
                font-weight: 600;
            }
            .tag-hook { background: #28a745; }
            .tag-quote { background: #17a2b8; }
            .tag-repeat { background: #dc3545; }
            .tag-ai { background: #6f42c1; }
            .tag-flow { background: #fd7e14; }
            .line {
                margin-bottom: 15px;
                padding: 10px;
                border-radius: 8px;
                background: rgba(255,255,255,0.05);
            }
            .recording {
                animation: pulse 1.5s infinite;
                color: red;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            #audio-preview {
                width: 100%;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>FlowState AI</h1>
            <p style="text-align: center; color: #aaa;">Freestyle → Song Converter</p>
            
            <div class="card">
                <h3>🎤 Input</h3>
                <textarea id="freestyle-input" placeholder="Type or record your freestyle..."></textarea>
                <div>
                    <button id="record-btn" class="btn">🎤 Record (10s)</button>
                    <button id="stop-btn" class="btn" style="display:none;">⏹ Stop</button>
                    <button id="analyze-btn" class="btn">🔍 Analyze</button>
                    <button id="generate-btn" class="btn">🎵 Generate Song</button>
                    <button id="reset-btn" class="btn">🔄 New Session</button>
                </div>
                <div id="recording-status" style="display: none;">
                    <span class="recording">●</span> Recording...
                </div>
                <audio id="audio-preview" controls style="display:none;"></audio>
            </div>
            
            <div class="card">
                <h3>📊 Analysis</h3>
                <div id="lines-container"></div>
            </div>
            
            <div class="card">
                <h3>🎶 Song Structure</h3>
                <div id="structure-container"></div>
                <div id="beat-suggestion" style="margin-top: 10px;"></div>
            </div>
            
            <div class="card">
                <h3>💾 Session</h3>
                <input type="text" id="artist-name" placeholder="Artist name (optional)">
                <button id="save-btn" class="btn">💾 Save Session</button>
                <button id="load-btn" class="btn">📂 Load Session</button>
                <div id="session-status"></div>
            </div>
        </div>

        <script>
            // Session Management
            let currentSessionId = null;
            
            // Audio Recording
            let mediaRecorder;
            let audioChunks = [];
            const recordBtn = document.getElementById('record-btn');
            const stopBtn = document.getElementById('stop-btn');
            const audioPreview = document.getElementById('audio-preview');
            
            recordBtn.addEventListener('click', startRecording);
            stopBtn.addEventListener('click', stopRecording);
            
            async function startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.start();
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    document.getElementById('recording-status').style.display = 'block';
                    recordBtn.style.display = 'none';
                    stopBtn.style.display = 'inline-block';
                    
                    // Auto-stop after 10 seconds
                    setTimeout(() => {
                        if (mediaRecorder.state === 'recording') {
                            stopRecording();
                        }
                    }, 10000);
                    
                } catch (err) {
                    alert("Microphone access denied. Please enable permissions.");
                }
            }
            
            function stopRecording() {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                document.getElementById('recording-status').style.display = 'none';
                recordBtn.style.display = 'inline-block';
                stopBtn.style.display = 'none';
            }
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append('audio', audioBlob);
                
                try {
                    const response = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (data.text) {
                        document.getElementById('freestyle-input').value = data.text;
                    }
                    
                    if (data.audio_preview) {
                        const audioBuffer = new Uint8Array(data.audio_preview.match(/.{1,2}/g).map(byte => parseInt(byte, 16))).buffer;
                        audioPreview.src = URL.createObjectURL(new Blob([audioBuffer], { type: 'audio/mp3' }));
                        audioPreview.style.display = 'block';
                    }
                    
                } catch (e) {
                    console.error("Error:", e);
                }
            };
            
            // Analysis and Generation
            document.getElementById('analyze-btn').addEventListener('click', analyzeLine);
            document.getElementById('generate-btn').addEventListener('click', generateStructure);
            document.getElementById('reset-btn').addEventListener('click', resetSession);
            document.getElementById('save-btn').addEventListener('click', saveSession);
            document.getElementById('load-btn').addEventListener('click', loadSession);
            
            async function analyzeLine() {
                const line = document.getElementById('freestyle-input').value.trim();
                if (!line) return;
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ line })
                });
                
                const data = await response.json();
                displayAnalysis(data);
                document.getElementById('freestyle-input').value = '';
            }
            
            function displayAnalysis(data) {
                const lineDiv = document.createElement('div');
                lineDiv.className = 'line';
                
                let tags = '';
                if (data.deja_vu) tags += '<span class="tag tag-repeat">Repeat</span>';
                if (data.hook) tags += '<span class="tag tag-hook">Hook</span>';
                if (data.key_quote) tags += '<span class="tag tag-quote">Key Line</span>';
                if (data.power_words.length > 0) {
                    tags += `<span class="tag tag-ai">Power: ${data.power_words.join(', ')}</span>`;
                }
                tags += `<span class="tag tag-flow">Flow: ${data.flow_score}/10</span>`;
                
                lineDiv.innerHTML = `${tags}<div>${data.line}</div>`;
                
                if (data.analysis) {
                    const analysisDiv = document.createElement('div');
                    analysisDiv.style = "font-size:0.8em;color:#bbb;margin-top:5px";
                    analysisDiv.textContent = data.analysis;
                    lineDiv.appendChild(analysisDiv);
                }
                
                document.getElementById('lines-container').appendChild(lineDiv);
            }
            
            async function generateStructure() {
                const response = await fetch('/generate');
                const data = await response.json();
                
                const container = document.getElementById('structure-container');
                container.innerHTML = `
                    <div style="white-space: pre-line; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;">
                        ${data.structure}
                    </div>
                    <p><strong>BPM:</strong> ${data.bpm} | <strong>Flow Type:</strong> ${data.flow_type || 'Not detected'}</p>
                `;
                
                // Check for matching beat
                const beatSuggestion = document.getElementById('beat-suggestion');
                beatSuggestion.innerHTML = '<p>Loading beat suggestion...</p>';
                
                const beatResponse = await fetch(`/beats/bpm_${data.bpm.split('-')[0]}.mp3`);
                if (beatResponse.ok) {
                    beatSuggestion.innerHTML = `
                        <p>🎧 Suggested beat: <a href="${beatResponse.url}" target="_blank">Play</a></p>
                        <audio controls src="${beatResponse.url}" style="width:100%"></audio>
                    `;
                } else {
                    beatSuggestion.innerHTML = '<p>No matching beat found for this BPM</p>';
                }
            }
            
            async function resetSession() {
                const response = await fetch('/reset', { method: 'POST' });
                const data = await response.json();
                currentSessionId = data.session_id;
                document.getElementById('lines-container').innerHTML = '';
                document.getElementById('structure-container').innerHTML = '';
                document.getElementById('freestyle-input').value = '';
                document.getElementById('session-status').innerHTML = '<p>New session started</p>';
            }
            
            async function saveSession() {
                const artist = document.getElementById('artist-name').value.trim() || null;
                const response = await fetch('/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ artist })
                });
                const data = await response.json();
                currentSessionId = data.session_id;
                document.getElementById('session-status').innerHTML = `
                    <p>Session saved! ID: ${data.session_id}</p>
                `;
            }
            
            async function loadSession() {
                const sessionId = prompt("Enter session ID to load:");
                if (!sessionId) return;
                
                const response = await fetch(`/load/${sessionId}`);
                if (response.ok) {
                    document.getElementById('lines-container').innerHTML = '';
                    document.getElementById('structure-container').innerHTML = '';
                    document.getElementById('session-status').innerHTML = '<p>Session loaded!</p>';
                    currentSessionId = sessionId;
                    
                    // Trigger analysis display for all lines
                    const lines = await fetch('/generate');
                    const structure = await lines.json();
                    document.getElementById('structure-container').innerHTML = `
                        <div style="white-space: pre-line; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;">
                            ${structure.structure}
                        </div>
                    `;
                } else {
                    alert("Session not found");
                }
            }
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # Create required directories
    os.makedirs('static/beats', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
