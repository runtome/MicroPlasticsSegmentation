"""
Convert COCO segmentation annotations to YOLO segmentation format.

YOLO seg format (per line in .txt):
    class_id x1 y1 x2 y2 ... xn yn
where coordinates are normalized by image width/height.
"""
import json
import os
import shutil
from pathlib import Path


def coco_to_yolo_seg(
    annotation_path: str,
    images_dir: str,
    output_dir: str,
    splits: dict,
    class_names: list = None,
) -> None:
    """
    Convert COCO annotations to YOLO segmentation format.

    Creates:
        output_dir/
            images/train/   ← symlinks or copies of images
            images/val/
            images/test/
            labels/train/   ← .txt label files
            labels/val/
            labels/test/
            dataset.yaml    ← YOLO dataset config
    """
    if class_names is None:
        class_names = ["Fiber", "Fragment", "Film"]

    with open(annotation_path) as f:
        coco = json.load(f)

    # category_id (1-indexed) -> YOLO class (0-indexed)
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    # Build sorted mapping: id -> 0-indexed class
    sorted_cat_ids = sorted(categories.keys())
    cat_id_to_yolo = {cid: i for i, cid in enumerate(sorted_cat_ids)}

    # Build lookups
    img_meta = {img["id"]: img for img in coco["images"]}
    img_fname_to_id = {img["file_name"]: img["id"] for img in coco["images"]}

    ann_by_image = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    output_dir = Path(output_dir)

    for split_name, file_names in [
        ("train", splits["train"]),
        ("val", splits["val"]),
        ("test", splits["test"]),
    ]:
        img_out_dir = output_dir / "images" / split_name
        lbl_out_dir = output_dir / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for fname in file_names:
            src_img = Path(images_dir) / fname
            dst_img = img_out_dir / fname

            # Copy or symlink image
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # Write label file
            img_id = img_fname_to_id.get(fname)
            if img_id is None:
                continue

            meta = img_meta[img_id]
            W, H = meta["width"], meta["height"]
            anns = ann_by_image.get(img_id, [])

            label_path = lbl_out_dir / (Path(fname).stem + ".txt")
            with open(label_path, "w") as f:
                for ann in anns:
                    yolo_class = cat_id_to_yolo[ann["category_id"]]
                    for seg in ann.get("segmentation", []):
                        if len(seg) < 6:
                            continue
                        pts = []
                        for i in range(0, len(seg), 2):
                            x = seg[i] / W
                            y = seg[i + 1] / H
                            pts.extend([f"{x:.6f}", f"{y:.6f}"])
                        line = str(yolo_class) + " " + " ".join(pts)
                        f.write(line + "\n")

    # Write dataset.yaml
    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"YOLO dataset written to {output_dir}")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['val'])} images")
    print(f"  Test:  {len(splits['test'])} images")
