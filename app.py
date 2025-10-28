import os
import json
import base64
import flask
import queue
import threading
import simple_websocket
from flask_sock import Sock
from flask_cors import CORS
from google.cloud import speech
from google.oauth2 import service_account
import google.generativeai as genai

# --- Flask setup ---
app = flask.Flask(__name__, template_folder="templates")
sock = Sock(app)
CORS(app)

# --- GOOGLE SPEECH-TO-TEXT CONFIG ---
b64_creds = os.getenv("GOOGLE_CREDENTIALS_B64")
if not b64_creds:
    raise Exception("❌ GOOGLE_CREDENTIALS_B64 not found")

creds_json = base64.b64decode(b64_creds).decode("utf-8")
creds_info = json.loads(creds_json)
credentials = service_account.Credentials.from_service_account_info(creds_info)
client = speech.SpeechClient(credentials=credentials)

# --- GEMINI CONFIG ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# --- ROUTES ---
@app.route("/")
def index():
    return flask.render_template("index.html")


# ✅ Speech-to-Text WebSocket (with real-time partial + final updates)
@sock.route("/audio")
def audio(ws):
    audio_q = queue.Queue()

    def receive_audio():
        while True:
            try:
                data = ws.receive()
            except simple_websocket.errors.ConnectionClosed:
                break
            if data is None:
                break
            audio_q.put(data)
        audio_q.put(None)

    threading.Thread(target=receive_audio, daemon=True).start()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code="en-US",
        enable_automatic_punctuation=False,  # ✅ disable punctuation cleanup
        use_enhanced=False,                  # ✅ use raw model
        model="default"                      # ✅ catches "uh", "um", etc.
)


    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,      # ✅ partial updates
        single_utterance=False
    )

    def request_generator():
        while True:
            chunk = audio_q.get()
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    try:
        responses = client.streaming_recognize(streaming_config, request_generator())
        for response in responses:
            for result in response.results:
                transcript = result.alternatives[0].transcript
                if result.is_final:
                    ws.send("[FINAL]" + transcript)
                else:
                    ws.send("[INTERIM]" + transcript)
    except Exception as e:
        print("🎧 Speech streaming error:", e)
        try:
            ws.send("[ERROR] Speech recognition error occurred.")
        except:
            pass
    finally:
        ws.close()


# ✅ Generate Interview Questions
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = flask.request.get_json()
        job_description = data.get("job_description", "")

        prompt = f"""
        You are an expert HR interviewer.
        Generate exactly 4 interview questions:
        - 2 general HR questions
        - 2 job-specific questions based on the following job description.

        Job Description:
        {job_description}

        Return only the questions in a numbered list format:
        1. ...
        2. ...
        3. ...
        4. ...
        """

        response = model.generate_content(prompt)
        questions_text = response.text.strip()

        questions = []
        for line in questions_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and ('.' in line[:4]):
                parts = line.split('.', 1)
                question = parts[1].strip() if len(parts) > 1 else line
            else:
                question = line
            questions.append(question)

        return flask.jsonify({"questions": questions})

    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500


# ✅ Evaluate Interview Answers
@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        data = flask.request.get_json()
        question = data.get("question", "")
        answer = data.get("answer", "")

        if not question or not answer:
            return flask.jsonify({"error": "Missing question or answer"}), 400

        prompt = f"""
        You are an expert interview evaluator.
        Evaluate the candidate's answer using the following 10 parameters.
        Each parameter should have a score from 1–10.
        Also calculate a total (sum out of 100).
        Provide a 2-3 sentence summary and 3-5 improvement tips.

        Parameters:
        1. Clarity
        2. Relevance
        3. Communication
        4. Confidence
        5. Structure
        6. Technical Depth
        7. Example Quality
        8. Conciseness
        9. Authenticity
        10. Impact

        Format response as pure JSON:
        {{
          "scores": {{
            "Clarity": 0,
            "Relevance": 0,
            "Communication": 0,
            "Confidence": 0,
            "Structure": 0,
            "Technical Depth": 0,
            "Example Quality": 0,
            "Conciseness": 0,
            "Authenticity": 0,
            "Impact": 0
          }},
          "total": 0,
          "summary": "short 2-3 sentence summary",
          "improvement_tips": [
            "tip 1",
            "tip 2",
            "tip 3"
          ]
        }}

        Question: {question}
        Answer: {answer}
        """

        response = model.generate_content(prompt)
        raw_text = getattr(response, "text", "")
        cleaned = raw_text.strip().replace("```json", "").replace("```", "")
        json_part = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]

        try:
            evaluation = json.loads(json_part)
        except Exception:
            evaluation = {"raw_evaluation": raw_text}

        return flask.jsonify({"evaluation": evaluation})

    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500


# ✅ Run Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
