import cv2


def main():
    cap = cv2.VideoCapture("Video.mp4")
    FPS = cap.get(cv2.CAP_PROP_FPS)
    print(FPS)

main()