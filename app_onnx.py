"""
ONNX Flask app — parallel version running on port 5001.
Uses ai_service_onnx instead of ai_service.
Your original app.py on port 5000 is untouched.
"""

import base64, os, time, uuid
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from ultralytics import YOLO
from groq import Groq
import json as json_lib
from dotenv import load_dotenv
load_dotenv()

import torch
# Patch torch.load for PyTorch 2.6+ compatibility
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# ONNX version of the RAG service
from ai_service_onnx import diagnose as rag_diagnose

app = Flask(__name__)
CORS(app)

# YOLO
model = YOLO("best.pt")
print(f"✅ Model loaded. Classes: {model.names}")

CLASS_NAMES = {0: "spaghetti", 1: "warping", 2: "under_extrusion", 3: "stringing"}
HISTORY = []

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)


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
@app.route("/", methods=["GET", "HEAD"])
def root():
    return jsonify({
        "service": "PrintGuard backend (ONNX)",
        "status": "ok",
        "endpoints": ["/status", "/upload_raw", "/detect", "/stream",
                      "/ai/chat", "/ai/explain", "/recommend"],
    })


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
        "version": "1.0.0-onnx",
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
            cv2.putText(err, "No camera", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
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
    import re
    a = analysis.lower()
    match = re.search(
        r"urgency\*{0,2}\s*[-–—:]\s*\*{0,2}\s*(high|moderate|medium|low)",
        a,
    )
    if not match:
        return "NOTIFY"
    word = match.group(1)
    if word == "high":
        return "STOP"
    if word in ("moderate", "medium"):
        return "NOTIFY"
    if word == "low":
        return "CONTINUE"
    return "NOTIFY"


def _split_analysis(analysis: str):
    a_low = analysis.lower()
    idx = a_low.find("suggested fix")
    if idx == -1:
        return analysis, "See analysis above."
    return analysis[:idx].strip(), analysis[idx:].strip()


# ── RAG endpoints ─────────────────────────────────────────────────
@app.route("/ai/chat", methods=["POST"])
def ai_chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided", "status": "error"}), 400
        user_messages = data.get("messages", [])
        if not user_messages:
            return jsonify({"error": "No messages provided", "status": "error"}), 400

        last_user = next((m for m in reversed(user_messages)
                          if m.get("role") == "user"), None)
        user_question = last_user.get("content", "") if last_user else ""
        if not user_question:
            return jsonify({"response": "No question provided.",
                            "status": "ok", "model_used": "onnx-rag+groq"})

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
                "model_used": "onnx-rag+groq",
                "validation": "no_class_detected",
            })

        print(f"📥 ONNX RAG chat: class={detected_class}, question={user_question[:60]}...")
        result = rag_diagnose(detected_class, 0.8, user_question=user_question)
        return jsonify({
            "response": result.get("analysis", "No response generated."),
            "status": "ok",
            "model_used": "onnx-rag+groq",
            "citations": result.get("citations", []),
            "sources": result.get("sources", []),
            "validation": result.get("validation", "unknown"),
        })
    except Exception as e:
        print(f"🔥 /ai/chat error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/ai/explain", methods=["POST"])
def ai_explain():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided", "status": "error"}), 400
        class_name = data.get("class_name", "unknown")
        confidence = data.get("confidence", 0)
        conf = float(confidence)
        if conf > 1.0:
            conf = conf / 100.0
        print(f"📥 ONNX RAG explain request: {class_name} ({conf:.2f})")
        result = rag_diagnose(class_name, conf)
        return jsonify({
            "response": result.get("analysis", "No analysis generated."),
            "status": "ok",
            "model_used": "onnx-rag+groq",
            "citations": result.get("citations", []),
            "sources": result.get("sources", []),
            "validation": result.get("validation", "unknown"),
        })
    except Exception as e:
        print(f"🔥 /ai/explain error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400
        class_name = data.get("class_name", "unknown")
        confidence = data.get("confidence", 0)
        conf = float(confidence)
        if conf > 1.0:
            conf = conf / 100.0
        print(f"📥 ONNX RAG recommendation: {class_name} ({conf:.2f})")
        result = rag_diagnose(class_name, conf)
        analysis = result.get("analysis", "")
        urgency = _extract_urgency(analysis)
        root_cause, suggested_fix = _split_analysis(analysis)
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)