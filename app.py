import base64, os, time, uuid
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from ultralytics import YOLO
from groq import Groq
import json as json_lib
import torch

# Patch torch.load for PyTorch 2.6+ compatibility (weights_only defaults to True)
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
from dotenv import load_dotenv

load_dotenv()  # must run BEFORE importing ai_service

from ai_service import diagnose as rag_diagnose  # RAG service

app = Flask(__name__)
CORS(app)
import traceback

@app.errorhandler(Exception)
def _log_exception(e):
    traceback.print_exc()
    return jsonify({"error": str(e), "status": "error"}), 500

# ── Load YOLO model ───────────────────────────────────────────────
model = YOLO("best.pt")
print(f"✅ Model loaded. Classes: {model.names}")

# Manual class mapping — keeps names readable even if the .pt file uses different ids
CLASS_NAMES = {
    0: "spaghetti",
    1: "warping",
    2: "under_extrusion",
    3: "stringing",
}

HISTORY = []

# ── GROQ client ───────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY)

# Legacy context dict — kept for backwards compatibility, not used by RAG endpoints
DEFECT_CONTEXTS = {
    "spaghetti": "Print detached from bed. Filament extruded into air. Usually unrecoverable.",
    "stringing": "Fine hairs between parts. Caused by high temp or retraction settings.",
    "warping": "Corners lifting from bed. Caused by cooling or poor bed adhesion.",
    "layer_shift": "Layers misaligned. Usually a loose belt or stepper issue.",
    "under_extrusion": "Missing material in layers. Clogged nozzle or incorrect flow rate.",
}


# ── Startup Groq sanity check ─────────────────────────────────────
def test_groq_api():
    try:
        print("🔍 Testing Groq API connection...")
        test_response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Reply with 'OK'"},
            ],
            max_tokens=50,
        )
        print(f"✅ Groq API is working! Response: {test_response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        return False


test_groq_api()


# ── Detection helpers ─────────────────────────────────────────────
def run_inference(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return [], None

    results = model(frame, conf=0.25, verbose=False)
    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = CLASS_NAMES.get(class_id, model.names[class_id])

        detections.append({
            "class": class_name,
            "confidence": round(float(box.conf[0]), 4),
            "x1": round(float(box.xyxy[0][0]), 1),
            "y1": round(float(box.xyxy[0][1]), 1),
            "x2": round(float(box.xyxy[0][2]), 1),
            "y2": round(float(box.xyxy[0][3]), 1),
        })

    annotated = results[0].plot()
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return detections, b64


# ── Detection routes ──────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    cap = cv2.VideoCapture(0)
    cam_ok = cap.isOpened()
    cap.release()
    last = HISTORY[-1]["timestamp"] if HISTORY else None
    return jsonify({
        "model_loaded": True,
        "camera_connected": cam_ok,
        "total_detections": len(HISTORY),
        "last_detection": last,
        "version": "1.0.0",
    })


@app.route("/upload_raw", methods=["POST"])
def upload_raw():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    image_bytes = request.files["image"].read()
    detections, b64 = run_inference(image_bytes)

    detection_id = str(uuid.uuid4())
    HISTORY.append({
        "id": detection_id,
        "timestamp": datetime.utcnow().isoformat(),
        "detections": detections,
    })
    if len(HISTORY) > 100:
        HISTORY.pop(0)

    return jsonify({
        "detections": detections,
        "annotated_image": b64,
        "detection_id": detection_id,
    })


@app.route("/detect", methods=["GET"])
def detect():
    if not HISTORY:
        return jsonify({"detections": [], "annotated_image": None})
    return jsonify(HISTORY[-1])


@app.route("/stream", methods=["GET"])
def stream():
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            err = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(err, "No camera", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buf = cv2.imencode(".jpg", err)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            return
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=0.25, verbose=False)
                annotated = results[0].plot()
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                time.sleep(0.1)
        finally:
            cap.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/alert", methods=["POST"])
def alert():
    data = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "detection_id": data.get("detection_id")})


# ── RAG helpers ───────────────────────────────────────────────────
def _extract_urgency(analysis: str) -> str:
    """
    Extract urgency from the LLM's own urgency statement.
    Looks for "Urgency" followed by High/Moderate/Medium/Low.
    Returns one of STOP / NOTIFY / CONTINUE.
    """
    a = analysis.lower()
    # Find the "urgency" keyword position
    idx = a.rfind("urgency")
    if idx == -1:
        return "NOTIFY"  # default when no urgency word found
    tail = a[idx:idx + 200]  # look at the text right after "urgency"
    if "high" in tail:
        return "STOP"
    if "moderate" in tail or "medium" in tail:
        return "NOTIFY"
    if "low" in tail:
        return "CONTINUE"
    return "NOTIFY"


def _split_analysis(analysis: str):
    """
    Split the RAG analysis into root cause + suggested fix sections.
    Falls back to (analysis, "See analysis above.") if no clear split.
    """
    a_low = analysis.lower()
    idx = a_low.find("suggested fix")
    if idx == -1:
        return analysis, "See analysis above."
    root_cause = analysis[:idx].strip()
    suggested_fix = analysis[idx:].strip()
    return root_cause, suggested_fix


# ── RAG-powered endpoints ─────────────────────────────────────────
@app.route("/ai/chat", methods=["POST"])
def ai_chat():
    """
    RAG-grounded chat. Infers defect class from conversation, then answers
    the user's question grounded in the knowledge base.
    Response shape unchanged: {response, status, model_used}.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided", "status": "error"}), 400

        user_messages = data.get("messages", [])
        if not user_messages:
            return jsonify({"error": "No messages provided", "status": "error"}), 400

        # Last user message is the current question
        last_user = next((m for m in reversed(user_messages)
                          if m.get("role") == "user"), None)
        user_question = last_user.get("content", "") if last_user else ""
        if not user_question:
            return jsonify({
                "response": "No question provided.",
                "status": "ok",
                "model_used": "rag+groq",
            })

        # Infer defect class from conversation history
        all_text = " ".join(m.get("content", "").lower() for m in user_messages)
        detected_class = None
        for cls in ("spaghetti", "warping", "under_extrusion", "stringing"):
            if cls.replace("_", " ") in all_text or cls in all_text:
                detected_class = cls
                break

        if not detected_class:
            return jsonify({
                "response": "I can only help with specific defects. Ask about spaghetti, warping, under_extrusion, or stringing.",
                "status": "ok",
                "model_used": "rag+groq",
                "validation": "no_class_detected",
            })

        print(f"📥 RAG chat: class={detected_class}, question={user_question[:60]}...")

        result = rag_diagnose(detected_class, 0.8, user_question=user_question)
        return jsonify({
            "response": result.get("analysis", "No response generated."),
            "status": "ok",
            "model_used": "rag+groq",
            "citations": result.get("citations", []),
            "sources": result.get("sources", []),
            "validation": result.get("validation", "unknown"),
        })

    except Exception as e:
        print(f"🔥 /ai/chat error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/ai/explain", methods=["POST"])
def ai_explain():
    """
    RAG-grounded explanation for a detection.
    Response shape unchanged: {response, status, model_used}.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided", "status": "error"}), 400

        class_name = data.get("class_name", "unknown")
        confidence = data.get("confidence", 0)

        # Normalize: mobile app sends 0-100, RAG expects 0.0-1.0
        conf = float(confidence)
        if conf > 1.0:
            conf = conf / 100.0

        print(f"📥 RAG explain request: {class_name} ({conf:.2f})")

        result = rag_diagnose(class_name, conf)
        return jsonify({
            "response": result.get("analysis", "No analysis generated."),
            "status": "ok",
            "model_used": "rag+groq",
            "citations": result.get("citations", []),
            "sources": result.get("sources", []),
            "validation": result.get("validation", "unknown"),
        })

    except Exception as e:
        print(f"🔥 /ai/explain error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    RAG-grounded recommendation. Returns the JSON shape the mobile app expects:
    {root_cause, suggested_fix, urgency_level} + extra fields (citations, sources, validation).
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400

        class_name = data.get("class_name", "unknown")
        confidence = data.get("confidence", 0)

        # Normalize confidence
        conf = float(confidence)
        if conf > 1.0:
            conf = conf / 100.0

        print(f"📥 RAG recommendation request: {class_name} ({conf:.2f})")

        result = rag_diagnose(class_name, conf)
        analysis = result.get("analysis", "")

        urgency = _extract_urgency(analysis)
        root_cause, suggested_fix = _split_analysis(analysis)

        print(f"✅ RAG recommendation: urgency={urgency}, validation={result.get('validation')}")

        return jsonify({
            "root_cause": root_cause or "No root cause available",
            "suggested_fix": suggested_fix or "No fix available",
            "urgency_level": urgency,
            "citations": result.get("citations", []),
            "sources": result.get("sources", []),
            "validation": result.get("validation", "unknown"),
        })

    except Exception as e:
        print(f"🔥 /recommend error: {e}")
        return jsonify({
            "root_cause": "Unable to generate recommendation",
            "suggested_fix": "Check server logs",
            "urgency_level": "NOTIFY",
        }), 200


# ── Debug route ───────────────────────────────────────────────────
@app.route("/debug/model", methods=["GET"])
def debug_model():
    return jsonify({
        "model_classes": model.names,
        "class_count": len(model.names),
        "all_classes": [{"id": k, "name": v} for k, v in model.names.items()],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)