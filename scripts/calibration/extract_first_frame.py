import cv2
import sys

def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video file."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_path, frame)
        print(f"First frame saved to {output_path}")
        print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("Error: Could not read first frame")

    cap.release()

if __name__ == "__main__":
    video_path = "data/Video/Batch_A1-A25/A1.mp4"
    output_path = "first_frame.jpg"
    extract_first_frame(video_path, output_path)
