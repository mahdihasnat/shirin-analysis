import os
import json
import re
import cv2
import argparse
import urllib.request
import numpy as np
import multiprocessing
from deepface import DeepFace
from sklearn.cluster import DBSCAN

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

def download_dnn_models(script_dir):
    """Download OpenCV DNN face detection models if not present."""
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    prototxt_path = os.path.join(models_dir, 'deploy.prototxt')
    caffemodel_path = os.path.join(models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    
    if not os.path.exists(prototxt_path):
        print("Downloading prototxt...")
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
        urllib.request.urlretrieve(url, prototxt_path)
    
    if not os.path.exists(caffemodel_path):
        print("Downloading model weights (this may take a moment)...")
        url = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
        urllib.request.urlretrieve(url, caffemodel_path)
    
    return prototxt_path, caffemodel_path

def detect_faces_dnn(image_path, net, conf_threshold=0.5):
    """
    Detects faces in an image using OpenCV DNN (ResNet SSD).
    Returns a list of bounding boxes [(x, y, w, h), ...].
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        height, width = img.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        net.setInput(blob)
        detections = net.forward()
        
        faces_list = []
        
        # Loop over detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * [width, height, width, height]
                (x1, y1, x2, y2) = box.astype("int")
                
                # Convert to x, y, w, h format
                x = int(max(0, x1))
                y = int(max(0, y1))
                w = int(max(0, x2 - x1))
                h = int(max(0, y2 - y1))
                
                faces_list.append((x, y, w, h))
        
        return faces_list
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

def extract_embeddings(image_path, faces):
    """
    Extracts face embeddings using DeepFace for a list of bounding boxes.
    Requires passing the pre-cropped face to avoid re-detecting.
    """
    embeddings = []
    
    img = cv2.imread(image_path)
    if img is None:
        return [None] * len(faces)
    
    for (x, y, w, h) in faces:
        try:
            # Add padding
            pad = max(int(min(w, h) * 0.1), 1)
            y1 = max(0, y - pad)
            y2 = min(img.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(img.shape[1], x + w + pad)
            
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                embeddings.append(None)
                continue

            # Generate embedding using Facenet (good balance of speed/accuracy)
            # enforce_detection=False because we already cropped exactly to the face
            embedding_objs = DeepFace.represent(img_path=face_crop, model_name="Facenet", enforce_detection=False)
            
            if embedding_objs and len(embedding_objs) > 0:
                embeddings.append(embedding_objs[0]["embedding"])
            else:
                embeddings.append(None)
        except Exception as e:
            print(f"Warning: Deepface failed to extract embedding for face in {image_path}: {e}")
            embeddings.append(None)
            
    return embeddings

def process_frame(args_tuple):
    """
    Worker function for multiprocessing. 
    args_tuple = (img_path, img_name, net_config, conf_threshold)
    net_config contains paths to prototxt and caffemodel since net object is not picklable.
    """
    img_path, img_name, prototxt_path, caffemodel_path, conf_threshold = args_tuple
    
    # Needs to init its own net locally per process
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    
    faces = detect_faces_dnn(img_path, net, conf_threshold)
    
    faces_data = [] # (face_idx, bbox, embedding)
    
    if len(faces) > 0:
        embeddings = extract_embeddings(img_path, faces)
        for face_idx, (bbox, emb) in enumerate(zip(faces, embeddings)):
            if emb is not None:
                faces_data.append((face_idx, bbox, emb))
                
    return img_name, len(faces), faces_data

def restrict_float_range(val):
    f = float(val)
    if f <= 0.0 or f >= 2.0:
        raise argparse.ArgumentTypeError(f"Value must be between 0.0 and 2.0. Got {val}")
    return f

def main():
    # Fix for TensorFlow multiprocessing deadlocks on Linux
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
        
    parser = argparse.ArgumentParser(description="Analyze frames and generate JSON annotations.")
    parser.add_argument("--frames-dir", default="../frame_extractor/output", help="Directory containing frames")
    parser.add_argument("--output-dir", default="../gallery_generator", help="Directory to save JSON files")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum detection confidence (0.0-1.0)")
    parser.add_argument("--eps", type=restrict_float_range, default=0.45, help="DBSCAN epsilon for face clustering (0.0 to 2.0 for cosine distance. Higher = fewer clusters)")
    
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

    # 2. Face Detection & Embeddings Extraction
    people_frames = []
    frames_by_count = {} # count -> list of frames
    
    # Store all faces globally for clustering
    # List of tuples: (filename, face_index_in_file, bbox, embedding)
    all_faces_data = []
    
    print("Loading OpenCV DNN face detector (ResNet SSD)...")
    prototxt_path, caffemodel_path = download_dnn_models(script_dir)
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    
    # Warm up DeepFace so we don't count its initialization in the loop
    print("Initializing DeepFace (Facenet)...")
    try:
        # Dummy call to force model download
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.represent(dummy_img, model_name="Facenet", enforce_detection=False)
    except:
        pass

    print(f"Analyzing {len(images)} frames using DNN detector and multiprocessing...")
    
    # Prepare arguments for multiprocessing
    tasks = []
    for img_name in images:
        img_path = os.path.join(abs_frames_dir, img_name)
        tasks.append((img_path, img_name, prototxt_path, caffemodel_path, args.confidence))
        
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_cores} cores.")

    with multiprocessing.Pool(processes=num_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(process_frame, tasks)):
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{len(images)} frames...")
                
            img_name, count, faces_data = result
            
            if count > 0:
                people_frames.append(img_name)
                
                if count not in frames_by_count:
                    frames_by_count[count] = []
                frames_by_count[count].append(img_name)
                
                for face_idx, bbox, emb in faces_data:
                    all_faces_data.append({
                        "filename": img_name,
                        "face_idx": face_idx,
                        "bbox": bbox,
                        "embedding": emb
                    })

    # 3. Clustering
    print(f"Extracted {len(all_faces_data)} valid face embeddings. Starting clustering...")
    faces_metadata = {} # filename -> list of dicts: {"box": [x,y,w,h], "person_id": ID}
    frames_by_person = {} # person_id -> set of filenames
    
    # Pre-populate faces_metadata with empty lists for all detected frames
    for f in people_frames:
        faces_metadata[f] = []

    if len(all_faces_data) > 0:
        # Convert embeddings to numpy array
        X = np.array([item["embedding"] for item in all_faces_data])
        
        # Use cosine distance for facial embeddings
        # eps parameter determines how close faces need to be to be the "same person"
        # 0.45 is a decent starting point for Facenet cosine distance
        dbscan = DBSCAN(eps=args.eps, min_samples=3, metric="cosine")
        labels = dbscan.fit_predict(X)
        
        # Map labels back to metadata
        for i, item in enumerate(all_faces_data):
            label = int(labels[i])
            filename = item["filename"]
            
            # Label -1 means noise (unidentified person)
            person_id = None if label == -1 else label
            
            faces_metadata[filename].append({
                "box": item["bbox"],
                "person_id": person_id
            })
            
            if person_id is not None:
                if person_id not in frames_by_person:
                    frames_by_person[person_id] = set()
                frames_by_person[person_id].add(filename)
    
    # Sort person frame sets
    for p_id in frames_by_person:
        frames_by_person[p_id] = sorted(list(frames_by_person[p_id]))

    # 4. Generate JSONs
    print("Generating JSON files...")
    
    rel_base_path = os.path.relpath(abs_frames_dir, abs_output_dir)

    # faces_metadata.json
    with open(os.path.join(abs_output_dir, "faces_metadata.json"), "w") as f:
        json.dump(faces_metadata, f, separators=(',', ':'))

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
            
    # frames_person_N.json (Identified individuals)
    person_filters = []
    # Identify top people by number of appearances (optional, sort descending)
    sorted_people = sorted(frames_by_person.items(), key=lambda x: len(x[1]), reverse=True)
    
    for rank, (p_id, frames) in enumerate(sorted_people):
        # We cap at generating top N filters to avoid spam if algorithm misbehaves
        # Let's say top 20 identified characters
        if rank >= 20: 
            # Still generate json, just not filter maybe, or just do all
            pass
            
        person_data = {
            "basePath": rel_base_path,
            "items": compress_to_ranges(frames)
        }
        filename = f"frames_person_{p_id}.json"
        with open(os.path.join(abs_output_dir, filename), "w") as f:
            json.dump(person_data, f, indent=2)
            
        person_filters.append({
            "id": f"person_{p_id}",
            "name": f"Person {p_id} ({len(frames)} frames)",
            "file": filename
        })

    # filters.json
    filters = [
        {"id": "all", "name": "All Frames", "file": "frames_all.json", "default": True},
        {"id": "people", "name": "People Detected (Any)", "file": "frames_people.json"}
    ]
    
    sorted_counts = sorted(frames_by_count.keys())
    for count in sorted_counts:
        name = f"{count} Person" if count == 1 else f"{count} People"
        filters.append({
            "id": f"people_{count}",
            "name": name,
            "file": f"frames_people_{count}.json"
        })
        
    if person_filters:
        # Add a visual separator grouping in the UI could be nice, but simple concat works
        filters.extend(person_filters)
    
    with open(os.path.join(abs_output_dir, "filters.json"), "w") as f:
        json.dump(filters, f, indent=2)

    num_identified = len(frames_by_person)
    unidentified = sum(1 for f in all_faces_data if f.get('person_id') is None) # -1 in dbscan
    
    print(f"Success! Detected {num_identified} distinct individuals across {len(people_frames)} frames.")

if __name__ == "__main__":
    main()
