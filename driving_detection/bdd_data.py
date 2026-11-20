"""BDD100K data loading shared by lab 2A (model.ipynb) and lab 2B (yolo.ipynb).

Both labs must use exactly the same images, so the split parameters (N_TRAIN, N_DEV, N_TEST, DEV_FRAC, SEED)
live here and load_splits() returns the same train / dev / test samples in both notebooks.
"""

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

KAGGLE_DATASET = "marquis03/bdd100k"
DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = DATA_DIR / "cache"
TRAIN_IMAGES = DATA_DIR / "train" / "images"
TRAIN_ANNOTATIONS = (
    DATA_DIR / "train" / "annotations" / "bdd100k_labels_images_train.json"
)
VAL_IMAGES = DATA_DIR / "val" / "images"
VAL_ANNOTATIONS = DATA_DIR / "val" / "annotations" / "bdd100k_labels_images_val.json"

N_TRAIN, N_DEV, N_TEST = 1000, 500, 1000  # random images per part; None = all of them
DEV_FRAC = 0.1  # share of train recording sessions held out for tuning
SEED = 0
IMG_W, IMG_H = 1280, 720

# "unknown" = light whose color is not specified in the annotation (trafficLightColor == "none"),
# it does not necessarily mean that the light is switched off.
CLASSES = [
    "traffic_sign",
    "traffic_light_red",
    "traffic_light_yellow",
    "traffic_light_green",
    "traffic_light_unknown",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# One color per class (RGB). Lights are drawn in the color they show.
CLASS_COLORS = {
    "traffic_sign": (0, 80, 255),
    "traffic_light_red": (255, 0, 0),
    "traffic_light_yellow": (255, 200, 0),
    "traffic_light_green": (0, 200, 0),
    "traffic_light_unknown": (150, 150, 150),
}


def _link_or_copy(src, dst):
    """Hard link src to dst (no extra disk space), or copy it when that is not possible. Existing files stay."""
    if not os.path.exists(dst):
        try:
            os.link(src, dst)
        except OSError:  # other disk or no hard links on this filesystem
            shutil.copy2(src, dst)
    return dst


def prepare_data(src=None):
    """Put BDD100K into DATA_DIR in the layout used here: {train,val}/images/*.jpg,
    {train,val}/annotations/bdd100k_labels_images_{train,val}.json and test/*.jpg.
    src: an unpacked copy with that layout; None = download KAGGLE_DATASET with kagglehub
    (cached in ~/.cache/kagglehub after the first time). Does nothing when the data is already there."""
    needed = [TRAIN_IMAGES, TRAIN_ANNOTATIONS, VAL_IMAGES, VAL_ANNOTATIONS]
    if all(p.exists() for p in needed):
        return DATA_DIR
    if src is None:
        import kagglehub

        src = kagglehub.dataset_download(KAGGLE_DATASET)
    src = Path(src)
    missing = [p for p in needed if not (src / p.relative_to(DATA_DIR)).exists()]
    if missing:
        raise FileNotFoundError(
            f"{src} does not have the expected layout, missing: {missing}"
        )
    for part in ("train", "val", "test"):
        if (src / part).exists():
            shutil.copytree(
                src / part,
                DATA_DIR / part,
                copy_function=_link_or_copy,
                dirs_exist_ok=True,
            )
    return DATA_DIR


def to_class(category, color):
    """Map a BDD100K (category, trafficLightColor) pair to one of CLASSES."""
    if category == "traffic sign":
        return "traffic_sign"
    return (
        f"traffic_light_{color}"
        if color in ("red", "yellow", "green")
        else "traffic_light_unknown"
    )


def iter_json_array(path, chunk_size=1 << 24):
    """Yield the elements of a top-level JSON array one at a time, in constant memory."""
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf = f.read(chunk_size)
        i = buf.index("[") + 1
        while True:
            while True:  # skip separators, refill buffer when exhausted
                while i < len(buf) and buf[i] in " \t\r\n,":
                    i += 1
                if i < len(buf):
                    break
                buf, i = f.read(chunk_size), 0
                if not buf:
                    return
            if buf[i] == "]":
                return
            try:
                obj, i = decoder.raw_decode(buf, i)
            except json.JSONDecodeError:  # element is cut by the chunk boundary
                more = f.read(chunk_size)
                if not more:
                    raise
                buf, i = buf[i:] + more, 0
                continue
            yield obj


def load_targets(json_path, cache_dir=CACHE_DIR):
    """One record per image: {name, attributes, objects: [[x1, y1, x2, y2, category, color], ...]}, signs and
    lights only. Parsed from the big JSON once, then read from a small cache file."""
    cache_path = Path(cache_dir) / f"{Path(json_path).stem}_targets.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    targets = []
    for entry in iter_json_array(json_path):
        objects = [
            [
                *(round(l["box2d"][k], 1) for k in ("x1", "y1", "x2", "y2")),
                l["category"],
                l.get("attributes", {}).get("trafficLightColor", "none"),
            ]
            for l in entry.get("labels", [])
            if "box2d" in l and l["category"] in ("traffic sign", "traffic light")
        ]
        targets.append(
            {
                "name": entry["name"],
                "attributes": entry.get("attributes", {}),
                "objects": objects,
            }
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")  # write + rename: no half-written cache
    tmp_path.write_text(json.dumps(targets))
    tmp_path.replace(cache_path)
    return targets


def split_train_dev(targets, dev_frac=DEV_FRAC, seed=SEED):
    """Split targets into (train, dev) by name prefix. Images with the same prefix probably come from one recording
    session (train has 42110 prefixes for 69863 images), so a per-image split would put near-identical scenes
    on both sides and inflate dev scores."""
    groups = sorted({t["name"].split("-")[0] for t in targets})
    random.Random(seed).shuffle(groups)
    dev_groups = set(groups[: round(len(groups) * dev_frac)])
    train = [t for t in targets if t["name"].split("-")[0] not in dev_groups]
    dev = [t for t in targets if t["name"].split("-")[0] in dev_groups]
    return train, dev


def to_samples(targets, images_dir, n=None, seed=SEED):
    """n random targets as {file_name: {"path", "boxes" (N, 4) xyxy, "labels" (N,) CLASSES ids, "attributes"}}."""
    if n is not None:
        targets = random.Random(seed).sample(targets, min(n, len(targets)))
    samples = {}
    for t in targets:
        labels = [CLASS_TO_ID[to_class(cat, color)] for *_, cat, color in t["objects"]]
        samples[t["name"]] = {
            "path": Path(images_dir) / t["name"],
            "boxes": np.array([o[:4] for o in t["objects"]], np.float32).reshape(-1, 4),
            "labels": np.array(labels, np.int64),
            "attributes": t["attributes"],
        }
    return samples


def load_splits(n_train=N_TRAIN, n_dev=N_DEV, n_test=N_TEST, seed=SEED):
    """The train / dev / test samples used in both labs. The official test split has no labels,
    so the labelled val split is the test set, and dev is held out from train by recording session."""
    train_part, dev_part = split_train_dev(load_targets(TRAIN_ANNOTATIONS), seed=seed)
    test_part = load_targets(VAL_ANNOTATIONS)
    return (
        to_samples(train_part, TRAIN_IMAGES, n_train, seed),
        to_samples(dev_part, TRAIN_IMAGES, n_dev, seed),
        to_samples(test_part, VAL_IMAGES, n_test, seed),
    )


def print_summary(name, samples):
    counts = Counter(CLASSES[c] for s in samples.values() for c in s["labels"])
    print(
        f"{name}: {len(samples)} images, {sum(counts.values())} boxes | "
        + ", ".join(f"{c.replace('traffic_', '')} {counts[c]}" for c in CLASSES)
    )


def load_image(sample):
    """The sample's image as an RGB array (H, W, 3)."""
    img = cv2.imread(str(sample["path"]))
    if img is None:
        raise FileNotFoundError(sample["path"])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_boxes(img, boxes, colors, thickness=2):
    """Draw xyxy boxes on a copy of the image. colors: one RGB tuple for all boxes or one per box."""
    if isinstance(colors, tuple):
        colors = [colors] * len(boxes)
    out = img.copy()
    for (x1, y1, x2, y2), color in zip(boxes, colors):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return out
