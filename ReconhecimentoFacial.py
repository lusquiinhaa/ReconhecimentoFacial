import cv2
import mediapipe as mp

# Configurações
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
DETECTION_CONFIDENCE = 0.5

def redimensionar_imagem(imagem):
    return cv2.resize(imagem, (RESIZE_WIDTH, RESIZE_HEIGHT))

def detectar_desenhar_pontos_faciais(imagem):
    resultado = reconhecimento_rosto.process(imagem)

    if resultado.multi_face_landmarks:
        for rosto_landmarks in resultado.multi_face_landmarks:
            desenho.draw_landmarks(imagem, rosto_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                   desenho.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                   desenho.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                                  )

def main():
    webcam = cv2.VideoCapture(0)

    while webcam.isOpened():
        validacao, frame = webcam.read()
        if not validacao:
            break

        imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagem = redimensionar_imagem(imagem)

        detectar_desenhar_pontos_faciais(imagem)

        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)
        cv2.imshow("Reconhecimento", imagem)

        if cv2.waitKey(5) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reconhecimento_rosto = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=DETECTION_CONFIDENCE, min_tracking_confidence=DETECTION_CONFIDENCE)
    desenho = mp.solutions.drawing_utils
    main()
