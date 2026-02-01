import cv2
import os

def extract_frames(video_path, output_dir, frames_per_second=1):
    """
    Extracts frames from a video at a specified rate.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where frames will be saved.
        frames_per_second (int): Number of frames to extract per second.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    # Calculate frame interval
    if frames_per_second <= 0:
        print("Error: frames_per_second must be positive")
        return

    frame_interval = int(fps / frames_per_second)
    if frame_interval == 0:
        frame_interval = 1 

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_filename, frame)
            # print(f"Saved {output_filename}") # Commented out to reduce noise, maybe print every 10 or 100
            if saved_count % 10 == 0:
                print(f"Saved {saved_count} frames...", end='\r')
            saved_count += 1
            
        frame_count += 1

    print(f"\nExtraction complete. Saved {saved_count} frames to {output_dir}")
    cap.release()

if __name__ == "__main__":
    video_file = "input/Shirin.Abbas.Kiarostami.2008.DVDRip.XViD.avi"
    output_folder = "output"
    
    # Get absolute paths to be safe
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, video_file)
    output_path = os.path.join(script_dir, output_folder)

    print(f"Extracting frames from {video_path} to {output_path}")
    extract_frames(video_path, output_path, frames_per_second=1)
