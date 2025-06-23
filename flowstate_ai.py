import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import openai
import re
import uuid
import datetime
import sqlite3
import json
import tempfile
import speech_recognition as sr
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from pydub import AudioSegment
from io import BytesIO
import logging
import psycopg2
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logger.warning("⚠️ Running in LOCAL-ONLY mode (no AI capabilities)")
    openai.api_key = "mock_key_for_dev"

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
        if os.getenv('DATABASE_URL'):  # Production - PostgreSQL
            url = urlparse(os.getenv('DATABASE_URL'))
            self.conn = psycopg2.connect(
                database=url.path[1:],
                user=url.username,
                password=url.password,
                host=url.hostname,
                port=url.port
            )
            logger.info("Connected to PostgreSQL database")
        else:  # Development - SQLite
            self.conn = sqlite3.connect('flowstate.db', check_same_thread=False)
            logger.info("Connected to SQLite database")
        
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
        try:
            self.c.execute('''INSERT INTO sessions VALUES (%s, %s, %s, %s)''',
                          (self.session_id, json.dumps(session_data), 
                           str(self.timestamp), artist_name))
            self.conn.commit()
            logger.info(f"Session saved: {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False
    
    def load_session(self, session_id):
        try:
            self.c.execute('''SELECT data FROM sessions WHERE id=%s''', (session_id,))
            data = self.c.fetchone()
            if data:
                session = json.loads(data[0])
                self.freestyle_history = session['history']
                self.hook_bank = defaultdict(list, session['hooks'])
                self.key_quotes = session['quotes']
                self.flow_type = session['flow_type']
                self.current_bpm = session['bpm']
                logger.info(f"Session loaded: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading session: {e}")
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
            logger.error(f"OpenAI Error: {e}")
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
            logger.error(f"OpenAI Error: {e}")
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
        beats_dir = os.path.join(app.static_folder, 'beats')
        if not os.path.exists(beats_dir):
            return None
        for f in os.listdir(beats_dir):
            if f.startswith(f"bpm_{target_bpm}"):
                return f"/static/beats/{f}"
        return None

# Initialize engine
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
    if engine.save_session(artist):
        return jsonify({"status": "saved", "session_id": engine.session_id})
    return jsonify({"error": "Failed to save session"}), 500

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
                "audio_preview": buffer.read().hex()
            })
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/beats/<path:filename>')
def serve_beat(filename):
    return send_from_directory('static/beats', filename)

def create_app():
    # Create required directories at startup
    if not os.path.exists('static/beats'):
        os.makedirs('static/beats', exist_ok=True)
    return app

# Production configuration
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app = create_app()
    app.run(host='0.0.0.0', port=port)
