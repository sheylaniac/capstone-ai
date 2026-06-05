import os
import base64
import cv2
import numpy as np
from deepface import DeepFace
import traceback
from fastapi import FastAPI, HTTPException, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Smart Digital Twin - AI Service",
    description="Dokumentasi API AI website Smart Digital Twin."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FaceAnalyzeRequest(BaseModel):
    image_base64: str

SECRET_TOKEN = os.getenv("SECRET_TOKEN_AI")

class AIService:
    @staticmethod
    def analyze_face_image(image_base64: str):
        try:
            cleaned_base64 = image_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "+")

            if "," in cleaned_base64:
                base64_data = cleaned_base64.split(",")[1]
            else:
                base64_data = cleaned_base64

            missing_padding = len(base64_data) % 4
            if missing_padding:
                base64_data += '=' * (4 - missing_padding)

            img_bytes = base64.b64decode(base64_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("[ERROR AI] OpenCV gagal merakit gambar (frame is None). Teks Base64 salah format.")
                raise ValueError("Format gambar tidak valid atau korup")

            results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion', 'age', 'gender'],
                detector_backend='retinaface',
                enforce_detection=False
            )
            result = results[0]

            emotion = result['dominant_emotion']
            age = result['age']
            confidence = float(result['emotion'][emotion] / 100)

            if emotion in ['sad', 'fear'] and confidence < 0.45:
                emotion = "neutral"
                confidence = float(result['emotion']['neutral'] / 100)

            gender = result.get('dominant_gender', 'unknown').lower()
            if 'woman' in gender or 'female' in gender:
                gender = "female"
            elif 'man' in gender or 'male' in gender:
                gender = "male"

            return {
                "predicted_age": int(age),
                "confidence_score": round(float(confidence), 4),
                "emotion": emotion,
                "gender": gender
            }

        except Exception as e:
            print("[CRITICAL ERROR DI AISERVICE HUGGINGFACE]")
            traceback.print_exc()
            raise ValueError(f"Proses AI gagal mengeksekusi gambar: {str(e)}")

class AnalyticsController:
    @staticmethod
    async def process_face_analysis(payload: FaceAnalyzeRequest, authorization: str = Header(None)):
        if authorization != f"Bearer {SECRET_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

        try:
            ai_result = AIService.analyze_face_image(payload.image_base64)
            return {
                "status": "success",
                "data": ai_result
            }
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=str(val_err))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

router = APIRouter(prefix="/api", tags=["AI Analytics"])
router.post("/face-detection")(AnalyticsController.process_face_analysis)

app.include_router(router)