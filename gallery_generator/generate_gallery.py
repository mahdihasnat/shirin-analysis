import os
import json
import re

def generate_gallery(frames_dir, output_file="frames.json"):
    """
    Generates a JSON file containing the list of images, optimized with ranges.
    
    Args:
        frames_dir (str): Path to the directory containing images.
        output_file (str): Path to the output JSON file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve frames_dir relative to the script location
    if not os.path.isabs(frames_dir):
        abs_frames_dir = os.path.normpath(os.path.join(script_dir, frames_dir))
    else:
        abs_frames_dir = frames_dir

    if not os.path.exists(abs_frames_dir):
        print(f"Error: Frames directory not found: {abs_frames_dir}")
        return

    # Get all image files
    images = sorted([f for f in os.listdir(abs_frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
    
    items = []
    
    # Regex to parse filenames: prefix, number, extension
    # e.g., "frame_0001.jpg" -> "frame_", "0001", ".jpg"
    pattern = re.compile(r'^(.*?)(\d+)(\.[a-zA-Z0-9]+)$')
    
    if not images:
        print("No images found.")
        return

    current_range = None

    for img in images:
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
                    items.append(current_range)
                
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
                items.append(current_range)
                current_range = None
            items.append(img)

    # Append the last range if exists
    if current_range:
        items.append(current_range)

    # Clean up single-item ranges to just be filenames? 
    # Optional, but keep it consistent for now. 
    # Actually, if start==end, we could convert back to string to save a tiny bit of client-side processing,
    # but having uniform objects is fine too. Let's stick to the logic.

    # Calculate relative path for basePath
    # frames_dir passed in is likely relative to PWD, but we want relative to index.html (script_dir)
    # If frames_dir was "../frame_extractor/output", we assume that's valid from index.html location too.
    
    # Let's strictly calculate relative path from script_dir -> abs_frames_dir
    rel_base_path = os.path.relpath(abs_frames_dir, script_dir)

    data = {
        "basePath": rel_base_path,
        "items": items
    }

    output_path = os.path.join(script_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Gallery data generated at: {output_path}")

if __name__ == "__main__":
    frames_directory = "../frame_extractor/output"
    generate_gallery(frames_directory)
