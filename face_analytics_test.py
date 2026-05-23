import cv2
from datetime import datetime, timezone
from deepface import DeepFace

# ganti jadi true fal
SEND_TO_BACKEND = False

#setup kamera
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

frame_counter       = 0
display_counter     = 0
display_duration    = 60
detection_frequency = 60

faces      = []
emotion    = ""
age        = 0
confidence = 0.0
# gender     = ""

#main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    smoothed_frame = cv2.medianBlur(frame, 5)
    gray_frame     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_counter += 1

    if frame_counter % detection_frequency == 0:
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            face_roi    = frame[y:y + h, x:x + w]
            captured_at = datetime.now(timezone.utc).isoformat()

            results = DeepFace.analyze(
                face_roi,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False
            )
            result = results[0]

            emotion    = result['dominant_emotion']
            age        = result['age']
            confidence = result['emotion'][emotion] / 100
            gender     = result['dominant_gender']

            display_counter = display_duration

            #cetak hasil
            if SEND_TO_BACKEND:
                import requests
                payload = {
                    "user_id":          1,
                    "predicted_age":    int(age),
                    "confidence_score": round(confidence, 4),
                    "emotion":          emotion,
                    "is_known_face":    False,
                    "source":           "webcam",
                    "captured_at":      captured_at
                }
                try:
                    #ganti fal
                    resp = requests.post(
                        "https://.....",
                        json=payload,
                        headers={"Authorization": "token..."},
                        timeout=5
                    )
                    print(f"[API] {resp.status_code} - {resp.json()}")
                except Exception as e:
                    print(f"[API ERROR] {e}")
            else:
                print(f"[TEST] Wajah terdeteksi!")
                print(f"  Emotion    : {emotion}")
                print(f"  Age        : {int(age)}")
                print(f"  Confidence : {round(confidence * 100, 1)}%")
                print(f"  Gender     : {gender}")
                print(f"  Captured   : {captured_at}")

    if display_counter > 0:
        display_counter -= 1
        for (x, y, w, h) in faces:
            cv2.rectangle(smoothed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(smoothed_frame, f"Emotion    : {emotion}",
                        (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(smoothed_frame, f"Age        : {int(age)}",
                        (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(smoothed_frame, f"Confidence : {round(confidence * 100, 1)}%",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(smoothed_frame, f"Gender     : {gender}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Real-time Face Analytics [TEST MODE]', smoothed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
