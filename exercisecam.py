import cv2

# 1. 사용할 분류기(Haar Cascade) 로드
# OpenCV 설치 시 내장된 기본 정면 얼굴 모델을 불러옵니다.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. 웹캠 연결 (0번은 기본 내장 카메라)
cap = cv2.VideoCapture(0)

print("📷 카메라를 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 속도와 정확도를 위해 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출 수행
    # scaleFactor: 이미지 크기 감소 비율, minNeighbors: 근처에 몇 개가 검출되어야 최종 인정할지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # 찾은 얼굴 위치에 사각형 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 결과 화면 표시
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()