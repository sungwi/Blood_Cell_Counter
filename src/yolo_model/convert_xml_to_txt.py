import os
import xml.etree.ElementTree as ET

def convert_bbox(size, box):
    """Convert bounding box from VOC format to YOLO format.
    """
    dw = 1.0 / size[0]  # width normalization
    dh = 1.0 / size[1]  # height normalization
    x_center = (box[0] + box[2]) / 2.0 * dw
    y_center = (box[1] + box[3]) / 2.0 * dh
    width = (box[2] - box[0]) * dw
    height = (box[3] - box[1]) * dh
    return x_center, y_center, width, height

def convert_annotation(xml_path, output_path, class_mapping=None):
    """Convert annotation from VOC XML format to YOLO txt format.
    """
    if class_mapping is None:
        class_mapping = {"cell": 0, "RBC": 0, "WBC": 0}
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return False
    except FileNotFoundError:
        print(f"XML file not found: {xml_path}")
        return False
    
    size_elem = root.find("size")
    if size_elem is None:
        print(f"Size element not found in {xml_path}")
        return False
    
    try:
        w = int(size_elem.find("width").text)
        h = int(size_elem.find("height").text)
    except (AttributeError, ValueError) as e:
        print(f"Error reading image dimensions from {xml_path}: {e}")
        return False
    
    # Process objects
    with open(output_path, "w") as out_file:
        found_objects = False
        
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None:
                continue
                
            cls = name_elem.text
            
            # Skip if class is not in mapping
            if cls not in class_mapping:
                print(f"Skipping unknown class '{cls}' in {xml_path}")
                continue
                
            cls_id = class_mapping[cls]
            
            # Get bounding box
            bbox_elem = obj.find("bndbox")
            if bbox_elem is None:
                continue
                
            try:
                xmin = float(bbox_elem.find("xmin").text)
                ymin = float(bbox_elem.find("ymin").text)
                xmax = float(bbox_elem.find("xmax").text)
                ymax = float(bbox_elem.find("ymax").text)
            except (AttributeError, ValueError) as e:
                print(f"Error reading bounding box from {xml_path}: {e}")
                continue
                
            # Convert and write to output file
            bb = convert_bbox((w, h), (xmin, ymin, xmax, ymax))
            out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")
            found_objects = True
            
        return found_objects

if __name__ == "__main__":
    class_mapping = {
        "RBC": 0,      # Red blood cells
        "cell": 0,     # General cell
        "WBC": 0,      # White blood cells
    }
    
    input_dir = "../data_processing/datasets/dataset_1/annotations"
    output_dir = "data_2/labels"
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    success_count = 0
    
    for xml_file in os.listdir(input_dir):
        if not xml_file.endswith(".xml"):
            continue
            
        processed_count += 1
        input_path = os.path.join(input_dir, xml_file)
        output_path = os.path.join(output_dir, xml_file.replace(".xml", ".txt"))
        
        print(f"Converting {xml_file}...")
        result = convert_annotation(input_path, output_path, class_mapping)
        
        if result:
            success_count += 1
    
    print(f"Conversion complete. Processed {processed_count} files, {success_count} successful.")