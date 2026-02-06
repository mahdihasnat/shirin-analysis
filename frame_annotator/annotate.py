import os
import json
import re
import cv2
import argparse

def compress_to_ranges(items):
    """
    Compresses a sorted list of filenames into range objects where possible.
    Target format: { "type": "range", "prefix": "frame_", "start": 1, "end": 100, "digits": 4, "extension": ".jpg" }
    """
    if not items:
        return []

    compressed_items = []
    pattern = re.compile(r'^(.*?)(\d+)(\.[a-zA-Z0-9]+)$')
    
    current_range = None

    for img in items:
        match = pattern.match(img)
        
        if match:
            prefix, num_str, ext = match.groups()
            num = int(num_str)
            digits = len(num_str)
            
            # Check if this file continues the current range
            if current_range and \
               current_range['prefix'] == prefix and \
               current_range['extension'] == ext and \
               current_range['digits'] == digits and \
               num == current_range['end'] + 1:
                
                current_range['end'] = num
            else:
                # Close previous range if it exists
                if current_range:
                    compressed_items.append(current_range)
                
                # Start new range
                current_range = {
                    "type": "range",
                    "prefix": prefix,
                    "start": num,
                    "end": num,
                    "digits": digits,
                    "extension": ext
                }
        else:
            # Non-matching file (no number sequence), treat as individual item
            if current_range:
                compressed_items.append(current_range)
                current_range = None
            compressed_items.append(img)

    # Append the last range if exists
    if current_range:
        compressed_items.append(current_range)
        
    return compressed_items

def detect_faces(image_path, face_cascade):
    """
    Detects faces in an image using OpenCV.
    Returns the number of detected faces.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Analyze frames and generate JSON annotations.")
    parser.add_argument("--frames-dir", default="../frame_extractor/output", help="Directory containing frames")
    parser.add_argument("--output-dir", default="../gallery_generator", help="Directory to save JSON files")
    args = parser.parse_args()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.frames_dir):
        abs_frames_dir = os.path.normpath(os.path.join(script_dir, args.frames_dir))
    else:
        abs_frames_dir = args.frames_dir
        
    if not os.path.isabs(args.output_dir):
        abs_output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))
    else:
        abs_output_dir = args.output_dir

    if not os.path.exists(abs_frames_dir):
        print(f"Error: Frames directory not found: {abs_frames_dir}")
        return

    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)

    # 1. Get all images
    print("Scanning frames...")
    images = sorted([f for f in os.listdir(abs_frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
    
    if not images:
        print("No images found.")
        return

    # 2. Face Detection
    people_frames = []
    frames_by_count = {} # count -> list of frames
    
    print("Loading face detector...")
    # Load pre-trained face detector from opencv directly
    # We need to find the path where cv2 keeps its data or download it.
    # cv2.data.haarcascades usually points to the installed location
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load haarcascade_frontalface_default.xml")
        return

    print(f"Analyzing {len(images)} frames for people (this may take a while)...")
    for i, img_name in enumerate(images):
        if i % 100 == 0:
            print(f"Processed {i}/{len(images)} frames...")
            
        img_path = os.path.join(abs_frames_dir, img_name)
        count = detect_faces(img_path, face_cascade)
        
        if count > 0:
            people_frames.append(img_name)
            
            if count not in frames_by_count:
                frames_by_count[count] = []
            frames_by_count[count].append(img_name)

    # 3. Generate JSONs
    print("Generating JSON files...")
    
    # Calculate relative path from output_dir to frames_dir for the JSON "basePath"
    # This is critical for the HTML to find the images relative to itself
    rel_base_path = os.path.relpath(abs_frames_dir, abs_output_dir)

    # frames_all.json
    all_data = {
        "basePath": rel_base_path,
        "items": compress_to_ranges(images)
    }
    with open(os.path.join(abs_output_dir, "frames_all.json"), "w") as f:
        json.dump(all_data, f, indent=2)

    # frames_people.json (Any people detected)
    people_data = {
        "basePath": rel_base_path,
        "items": compress_to_ranges(people_frames)
    }
    with open(os.path.join(abs_output_dir, "frames_people.json"), "w") as f:
        json.dump(people_data, f, indent=2)
        
    # frames_people_N.json (Specific counts)
    for count, frames in frames_by_count.items():
        count_data = {
            "basePath": rel_base_path,
            "items": compress_to_ranges(frames)
        }
        filename = f"frames_people_{count}.json"
        with open(os.path.join(abs_output_dir, filename), "w") as f:
            json.dump(count_data, f, indent=2)

    # filters.json
    filters = [
        {"id": "all", "name": "All Frames", "file": "frames_all.json", "default": True},
        {"id": "people", "name": "People Detected (Any)", "file": "frames_people.json"}
    ]
    
    # Sort counts for display
    sorted_counts = sorted(frames_by_count.keys())
    for count in sorted_counts:
        name = f"{count} Person" if count == 1 else f"{count} People"
        filters.append({
            "id": f"people_{count}",
            "name": name,
            "file": f"frames_people_{count}.json"
        })
    
    with open(os.path.join(abs_output_dir, "filters.json"), "w") as f:
        json.dump(filters, f, indent=2)

    print("Success! Annotations generated.")

if __name__ == "__main__":
    main()
