import os
from pathlib import Path

PERSON_CLASSES = {1, 2}

def convert(visdrone_ann_dir, output_label_dir, img_width, img_height):
    os.makedirs(output_label_dir, exist_ok=True)

    for ann_file in Path(visdrone_ann_dir).glob("*.txt"):
        out_lines = []
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split(",")
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                category = int(parts[5])

                if category not in PERSON_CLASSES:
                    continue
                cx = (x + w / 2) / img_width
                cy = (y + h / 2) / img_height
                nw = w / img_width
                nh = h / img_height

                out_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        out_path = Path(output_label_dir) / ann_file.name
        with open(out_path, "w") as f:
            f.write("\n".join(out_lines))

convert("C:/Users/prann/Desktop/VisDrone2019-DET-train/annotations", "dataset/labels/train", 1920, 1080)
convert("C:/Users/prann/Desktop/VisDrone2019-DET-val/annotations",   "dataset/labels/val",   1920, 1080)