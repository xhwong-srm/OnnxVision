"""Interactive ROI classification dataset builder.

The editor works on full source images.  ROIs can be drawn, moved with the
right mouse button, multi-selected with Ctrl, learned as a template pattern,
auto-populated to image edges, classified with an embedded ONNX classifier,
and exported as class-folder crops, optionally split into train/val/test.
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QBrush, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QGraphicsItem,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QInputDialog, QLabel, QListWidget, QMainWindow, QMessageBox,
    QProgressDialog, QPushButton, QScrollArea, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

SUPPORTED_SUFFIXES = {".bmp", ".dng", ".gif", ".heic", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
BAD = set('<>:"/\\|?*')
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


class ExportCancelled(Exception):
    pass


def valid_name(value: str, kind: str = "Name") -> str:
    value = value.strip()
    if not value or value in {".", ".."} or any(c in BAD for c in value) or value.endswith((".", " ")):
        raise ValueError(f"{kind} is empty or contains invalid Windows filename characters.")
    if value.split(".", 1)[0].upper() in RESERVED:
        raise ValueError(f"'{value}' is a reserved Windows filename.")
    return value


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).copy()


def source_image_id(path: Path, length: int = 12) -> str:
    """Return a stable ID that survives moving or copying the source image."""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def find_entries_by_source_id(entries, value):
    """Find loaded images by the 12-character content hash used in exports."""
    normalized = value.strip().lower()
    if len(normalized) != 12:
        raise ValueError("Enter exactly 12 hexadecimal characters.")
    try:
        int(normalized, 16)
    except ValueError as error:
        raise ValueError("The image hash may contain only 0-9 and A-F.") from error
    return [
        index for index, entry in enumerate(entries)
        if source_image_id(entry.path).lower() == normalized
    ]


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    area = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - area
    return area / union if union else 0.0


def find_pattern_matches(image, template, threshold, suppression_iou=0.30, maximum_matches=500):
    """Find distinct anchor matches using the same NMS policy as the detection builder."""
    template_height, template_width = template.shape
    if image.shape[0] < template_height or image.shape[1] < template_width:
        raise ValueError("image is smaller than the learned anchor pattern")
    scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(np.isfinite(scores) & (scores >= threshold))
    candidates = sorted(((int(x), int(y), float(scores[y, x])) for x, y in zip(xs, ys)), key=lambda item: item[2], reverse=True)
    selected = []
    for x, y, score in candidates:
        rect = (x, y, template_width, template_height)
        if any(iou(rect, (old_x, old_y, template_width, template_height)) > suppression_iou for old_x, old_y, _ in selected):
            continue
        selected.append((x, y, score))
        if len(selected) >= maximum_matches:
            break
    return selected


def randomize_sides(rect, size, ranges, rng):
    x, y, w, h = rect; iw, ih = size
    values = [round(base * rng.uniform(*sorted((float(lo), float(hi)))) / 100) for base, (lo, hi) in zip((w, w, h, h), ranges)]
    x1, y1 = max(0, x - values[0]), max(0, y - values[2])
    x2, y2 = min(iw, x + w + values[1]), min(ih, y + h + values[3])
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def force_ratio(rect, ratio: float, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Resize around the ROI center while keeping it inside the image."""
    if ratio <= 0:
        return rect
    x, y, w, h = rect; iw, ih = size
    if w / h > ratio:
        h = max(1, round(w / ratio))
    else:
        w = max(1, round(h * ratio))
    w, h = min(w, iw), min(h, ih)
    return max(0, min(iw - w, round(x + (rect[2] - w) / 2))), max(0, min(ih - h, round(y + (rect[3] - h) / 2))), w, h


def median_gap(rois: list[tuple[int, int, int, int]]) -> tuple[float, float]:
    ordered = sorted(rois, key=lambda r: r[0] + r[2] / 2)
    if len(ordered) < 2:
        raise ValueError("At least two ROIs are required to determine the gap median.")
    centers = [(x + w / 2, y + h / 2) for x, y, w, h in ordered]
    return (float(np.median(np.diff([p[0] for p in centers]))), float(np.median(np.diff([p[1] for p in centers]))))


def populate_edges(rois, size, ranges=((0, 0),) * 4, rng=None):
    """Extend a horizontal ROI sequence to both boundaries using median gaps."""
    rng = rng or random.Random()
    ordered = sorted(rois, key=lambda r: r[0] + r[2] / 2)
    gap_x, gap_y = median_gap(ordered)
    if gap_x <= 0: raise ValueError("ROIs need increasing horizontal centers.")
    iw, ih = size; result = list(ordered)
    for source, direction in ((ordered[0], -1), (ordered[-1], 1)):
        cx, cy = source[0] + source[2] / 2, source[1] + source[3] / 2
        step = 1
        while True:
            ncx, ncy = cx + direction * gap_x * step, cy + direction * gap_y * step
            if not (0 <= ncx < iw and 0 <= ncy < ih): break
            base = (round(ncx - source[2] / 2), round(ncy - source[3] / 2), source[2], source[3])
            # Do not clamp a candidate into the image.  If the complete ROI
            # does not fit, the edge has been reached and this direction ends.
            if base[0] < 0 or base[1] < 0 or base[0] + base[2] > iw or base[1] + base[3] > ih: break
            candidate = randomize_sides(base, size, ranges, rng)
            if any(iou(candidate, old) > .5 for old in result): break
            result.append(candidate); step += 1
    return sorted(result, key=lambda r: r[0] + r[2] / 2)


def split_counts(count, ratios):
    exact = [count * r for r in ratios]; result = [int(x) for x in exact]
    for i in sorted(range(3), key=lambda i: exact[i] - result[i], reverse=True)[:count - sum(result)]: result[i] += 1
    return result


def split_samples(samples, ratios, rng, group_duplicates=False, group_by_source=False):
    """Split labeled ROI samples, optionally grouping duplicates or source images."""
    split_names = ("train", "val", "test")
    splits = {name: [] for name in split_names}
    if group_duplicates or group_by_source:
        sample_groups = {}
        for sample in samples:
            entry, roi, index = sample
            if group_by_source:
                key = entry.path.resolve()
            elif roi.duplicate_group is not None:
                key = (entry.path.resolve(), roi.duplicate_group)
            else:
                key = (entry.path.resolve(), index)
            sample_groups.setdefault(key, []).append(sample)
        groups = list(sample_groups.values())
        rng.shuffle(groups)
        counts = split_counts(len(groups), ratios)
        offset = 0
        for name, count in zip(split_names, counts):
            for group in groups[offset:offset + count]:
                splits[name].extend(group)
            offset += count
        return splits

    by_class = {}
    for sample in samples:
        by_class.setdefault(sample[1].class_id, []).append(sample)
    for group in by_class.values():
        rng.shuffle(group)
        counts = split_counts(len(group), ratios)
        offset = 0
        for name, count in zip(split_names, counts):
            splits[name].extend(group[offset:offset + count])
            offset += count
    return splits


def split_entries(entries, ratios, rng):
    """Split whole source images so annotations from one image never leak."""
    shuffled = list(entries)
    rng.shuffle(shuffled)
    counts = split_counts(len(shuffled), ratios)
    result = {}
    offset = 0
    for name, count in zip(("train", "val", "test"), counts):
        result[name] = shuffled[offset:offset + count]
        offset += count
    return result


def export_coco_detection_dataset(entries, classes, output, ratios, seed=42, progress=None):
    """Write full images and bounding boxes in COCO detection format."""
    splits = split_entries(entries, ratios, random.Random(seed))
    categories = [
        {"id": class_id + 1, "name": name, "supercategory": ""}
        for class_id, name in enumerate(classes)
    ]
    total_images = 0
    total_annotations = 0
    image_total = sum(len(group) for group in splits.values())
    processed = 0

    for split_name, split_entries_for_name in splits.items():
        if not split_entries_for_name:
            continue
        image_dir = output / "images" / split_name
        annotation_dir = output / "annotations"
        image_dir.mkdir(parents=True, exist_ok=True)
        annotation_dir.mkdir(parents=True, exist_ok=True)
        coco_images = []
        coco_annotations = []

        for image_id, entry in enumerate(split_entries_for_name, start=1):
            file_name = f"{source_image_id(entry.path)}_{image_id:06d}.png"
            image = load_image(entry.path)
            image.save(image_dir / file_name, compress_level=1)
            coco_images.append({
                "id": image_id,
                "file_name": f"images/{split_name}/{file_name}",
                "width": image.width,
                "height": image.height,
            })
            for roi in entry.rois:
                if roi.class_id is None or not 0 <= roi.class_id < len(classes):
                    continue
                x, y, width, height = roi.rect
                annotation_id = len(coco_annotations) + 1
                coco_annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": roi.class_id + 1,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                })
            processed += 1
            if progress:
                progress(processed, image_total, f"Exporting {split_name}: {entry.path.name}")

        document = {
            "info": {"description": "ROI Dataset Builder COCO detection export"},
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        }
        with (annotation_dir / f"instances_{split_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        total_images += len(coco_images)
        total_annotations += len(coco_annotations)

    return total_images, total_annotations


def load_coco_detection_dataset(dataset_root):
    """Load COCO split files into editable entries and a shared class list."""
    dataset_root = Path(dataset_root).resolve()
    annotation_files = sorted((dataset_root / "annotations").glob("instances_*.json"))
    if not annotation_files:
        annotation_files = sorted(dataset_root.glob("**/instances_*.json"))
    if not annotation_files:
        raise ValueError("No COCO instances_*.json files were found.")

    documents = []
    classes = []
    class_lookup = {}
    for annotation_path in annotation_files:
        with annotation_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        category_map = {}
        for category in document.get("categories", []):
            name = str(category.get("name", "")).strip()
            if not name:
                continue
            key = name.casefold()
            if key not in class_lookup:
                class_lookup[key] = len(classes)
                classes.append(name)
            category_map[category["id"]] = class_lookup[key]
        documents.append((annotation_path, document, category_map))

    entries_by_path = {}
    for annotation_path, document, category_map in documents:
        annotations_by_image = defaultdict(list)
        for annotation in document.get("annotations", []):
            annotations_by_image[annotation.get("image_id")].append(annotation)
        for image_record in document.get("images", []):
            relative_path = Path(str(image_record.get("file_name", "")))
            candidates = (
                dataset_root / relative_path,
                annotation_path.parent / relative_path,
                dataset_root / "images" / relative_path,
            )
            image_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if image_path is None:
                raise FileNotFoundError(f"COCO image was not found: {relative_path}")
            if image_path not in entries_by_path:
                image = load_image(image_path)
                entries_by_path[image_path] = Entry(image_path, image.size)
            entry = entries_by_path[image_path]
            image_width, image_height = entry.size
            for annotation in annotations_by_image.get(image_record.get("id"), []):
                class_id = category_map.get(annotation.get("category_id"))
                bbox = annotation.get("bbox")
                if class_id is None or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x, y, width, height = (round(float(value)) for value in bbox)
                x = max(0, min(x, image_width - 1))
                y = max(0, min(y, image_height - 1))
                width = max(1, min(width, image_width - x))
                height = max(1, min(height, image_height - y))
                entry.rois.append(ROI((x, y, width, height), class_id))

    return list(entries_by_path.values()), classes


@dataclass
class ROI:
    rect: tuple[int, int, int, int]
    class_id: int | None = None
    confidence: float | None = None
    duplicate_group: int | None = None


@dataclass
class Entry:
    path: Path
    size: tuple[int, int]
    rois: list[ROI] = field(default_factory=list)
    pattern_matches: dict[str, list[tuple[int, int, int, int, float]]] = field(default_factory=dict)


@dataclass
class Pattern:
    name: str
    reference: Path
    rect: tuple[int, int, int, int]
    template: np.ndarray
    rois: list[ROI]


def best_pattern_matches(image, patterns, threshold, suppression_iou, maximum_matches):
    """Return matches only from the pattern with the best score in an image."""
    winner = None
    winner_matches = []
    for pattern in patterns:
        try:
            matches = find_pattern_matches(
                image, pattern.template, threshold, suppression_iou, maximum_matches
            )
        except ValueError:
            continue
        if matches and (not winner_matches or matches[0][2] > winner_matches[0][2]):
            winner = pattern
            winner_matches = matches
    return winner, winner_matches


def sort_rois_reading_order(rois):
    """Sort approximate horizontal rows top-to-bottom, then left-to-right."""
    if len(rois) < 2:
        return list(rois)
    tolerance = max(1.0, float(np.median([roi.rect[3] for roi in rois])) * 0.5)
    rows = []
    for roi in sorted(rois, key=lambda item: (item.rect[1] + item.rect[3] / 2, item.rect[0])):
        center_y = roi.rect[1] + roi.rect[3] / 2
        if not rows or abs(center_y - rows[-1][0]) > tolerance:
            rows.append([center_y, [roi]])
        else:
            rows[-1][1].append(roi)
            rows[-1][0] = sum(item.rect[1] + item.rect[3] / 2 for item in rows[-1][1]) / len(rows[-1][1])
    return [roi for _, row in rows for roi in sorted(row, key=lambda item: (item.rect[0], item.rect[1]))]


class CollapsibleSection(QWidget):
    def __init__(self, title, content_layout, expanded=False):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content = QWidget()
        self.content.setLayout(content_layout)
        self.content.setVisible(expanded)
        self.toggle.toggled.connect(self.set_expanded)
        outer.addWidget(self.toggle)
        outer.addWidget(self.content)

    def set_expanded(self, expanded):
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(expanded)


class RoiItem(QGraphicsRectItem):
    HANDLE_SIZE = 10

    def __init__(self, rect, bounds, label, changed, finished, ratio):
        super().__init__(QRectF(0, 0, rect[2], rect[3])); self.setPos(rect[0], rect[1]); self.bounds = bounds; self.label = label; self.changed = changed; self.finished = finished; self.ratio = ratio; self.resize_corner = None; self.resize_anchor = None
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable); self.setAcceptHoverEvents(True); self.setZValue(2)
        self.setPen(QPen(Qt.GlobalColor.yellow, 2)); self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
    def scene_rect(self): return QRectF(self.pos(), self.rect().size())
    def boundingRect(self): return self.rect().adjusted(-5, -5, 5, 5)
    def corner_at(self, position):
        width, height = self.rect().width(), self.rect().height()
        corners = {
            "top_left": QPointF(0, 0),
            "top_right": QPointF(width, 0),
            "bottom_left": QPointF(0, height),
            "bottom_right": QPointF(width, height),
        }
        return next((name for name, point in corners.items() if abs(position.x()-point.x())<=self.HANDLE_SIZE and abs(position.y()-point.y())<=self.HANDLE_SIZE), None)
    def paint(self, painter, option, widget=None):
        painter.setPen(QPen(Qt.GlobalColor.cyan if self.isSelected() else Qt.GlobalColor.yellow, 3))
        painter.drawRect(self.rect())
        metrics = painter.fontMetrics()
        badge_height = min(self.rect().height(), metrics.height() + 6)
        available_width = max(0, self.rect().width() - 6)
        text = metrics.elidedText(self.label, Qt.TextElideMode.ElideRight, int(available_width))
        badge_width = min(self.rect().width(), metrics.horizontalAdvance(text) + 8)
        badge = QRectF(self.rect().x(), self.rect().y(), badge_width, badge_height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 190)))
        painter.drawRoundedRect(badge, 3, 3)
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.drawText(badge.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter, text)
        if self.isSelected():
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            half = self.HANDLE_SIZE / 2
            for point in (self.rect().topLeft(), self.rect().topRight(), self.rect().bottomLeft(), self.rect().bottomRight()):
                painter.drawRect(QRectF(point.x()-half, point.y()-half, self.HANDLE_SIZE, self.HANDLE_SIZE))
    def hoverMoveEvent(self, event):
        corner = self.corner_at(event.pos())
        if corner in {"top_left", "bottom_right"}: self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif corner: self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        else: self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)
    def mousePressEvent(self, event):
        if event.button()==Qt.MouseButton.LeftButton:
            self.resize_corner=self.corner_at(event.pos())
            if self.resize_corner:
                rect=self.scene_rect()
                anchor_x=rect.right() if "left" in self.resize_corner else rect.left()
                anchor_y=rect.bottom() if "top" in self.resize_corner else rect.top()
                self.resize_anchor=QPointF(anchor_x,anchor_y)
                event.accept()
                return
        super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if self.resize_corner and self.resize_anchor is not None:
            left_side="left" in self.resize_corner
            top_side="top" in self.resize_corner
            maximum_width=self.resize_anchor.x() if left_side else self.bounds.width()-self.resize_anchor.x()
            maximum_height=self.resize_anchor.y() if top_side else self.bounds.height()-self.resize_anchor.y()
            width=max(2.0,min(abs(event.scenePos().x()-self.resize_anchor.x()),maximum_width))
            height=max(2.0,min(abs(event.scenePos().y()-self.resize_anchor.y()),maximum_height))
            ratio=float(self.ratio())
            if ratio>0:
                if width/height>ratio: height=width/ratio
                else: width=height*ratio
                scale=min(1.0,maximum_width/width,maximum_height/height)
                width*=scale; height*=scale
            x=self.resize_anchor.x()-width if left_side else self.resize_anchor.x()
            y=self.resize_anchor.y()-height if top_side else self.resize_anchor.y()
            self.prepareGeometryChange()
            self.setPos(x,y)
            self.setRect(0,0,width,height)
            self.changed(self.scene_rect())
            event.accept()
            return
        super().mouseMoveEvent(event); r = self.scene_rect(); self.setPos(max(0, min(r.x(), self.bounds.width() - r.width())), max(0, min(r.y(), self.bounds.height() - r.height()))); self.changed(self.scene_rect())
    def mouseReleaseEvent(self, event):
        if self.resize_corner:
            self.changed(self.scene_rect()); self.resize_corner=None; self.resize_anchor=None; self.finished(); event.accept(); return
        self.changed(self.scene_rect()); super().mouseReleaseEvent(event); self.finished()


class ImageView(QGraphicsView):
    def __init__(self, created, finished):
        super().__init__(); self.created = created; self.finished = finished; self.bounds = QRectF(); self.drawing = False; self.start = QPointF(); self.draft = None; self.right_drag = False; self.last = QPointF(); self.setDragMode(QGraphicsView.DragMode.NoDrag)
    def mousePressEvent(self, event):
        p = self.mapToScene(event.position().toPoint()); item = self.itemAt(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and self.bounds.contains(p) and not isinstance(item, RoiItem):
            self.drawing = True; self.start = p; self.draft = QGraphicsRectItem(); self.draft.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)); self.scene().addItem(self.draft); event.accept(); return
        if event.button() == Qt.MouseButton.RightButton:
            self.right_drag = True; self.last = p; event.accept(); return
        super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        p = self.mapToScene(event.position().toPoint())
        if self.drawing and self.draft: self.draft.setRect(QRectF(self.start, p).normalized().intersected(self.bounds)); event.accept(); return
        if self.right_drag:
            delta = p - self.last
            for item in self.scene().selectedItems():
                if isinstance(item, RoiItem): item.moveBy(delta.x(), delta.y()); item.changed(item.scene_rect())
            self.last = p; event.accept(); return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event):
        if self.drawing and self.draft:
            rect = self.draft.rect(); self.scene().removeItem(self.draft); self.draft = None; self.drawing = False
            if rect.width() >= 2 and rect.height() >= 2: self.created(rect)
            event.accept(); return
        if self.right_drag: self.right_drag = False; self.finished(); event.accept(); return
        super().mouseReleaseEvent(event)
    def wheelEvent(self, event): self.scale(1.2 if event.angleDelta().y() > 0 else 1 / 1.2, 1.2 if event.angleDelta().y() > 0 else 1 / 1.2)


class MainWindow(QMainWindow):
    def __init__(self, initial_model=None):
        super().__init__(); self.setWindowTitle("Classification ROI Dataset Builder"); self.resize(1450, 850); self.entries=[]; self.classes=[]; self.patterns=[]; self.current=-1; self.items=[]; self.pattern_items=[]; self.roi_clipboard=[]; self.learning_pattern=False; self.pattern_roi_indices=[]; self.model_path=Path(initial_model).resolve() if initial_model else None; self.session=None; self.input_name=None; self.kind=None; self.names={}
        root=QWidget(); self.setCentralWidget(root); layout=QHBoxLayout(root)
        left=QVBoxLayout(); left.addWidget(QLabel("Images")); self.images=QListWidget(); self.images.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection); self.images.currentRowChanged.connect(self.select); left.addWidget(self.images,1)
        for text, fn in (("Add images...",self.add_images),("Add folder...",self.add_folder),("Load COCO detection dataset...",self.load_detection),("Locate image by 12-char hash...",self.locate_image_by_hash),("Remove selected image(s)",self.remove_selected_images),("Remove repeated images",self.remove_repeated_images),("Delete ROIs on current",self.clear_current),("Delete ROIs on all",self.clear_all),("Clear images",self.clear_images)):
            b=QPushButton(text); b.clicked.connect(fn); left.addWidget(b)
        layout.addLayout(left,1)
        center=QVBoxLayout(); self.scene=QGraphicsScene(); self.pixmap=QGraphicsPixmapItem(); self.scene.addItem(self.pixmap); self.view=ImageView(self.created,self.sort_current_rois); self.view.setScene(self.scene); center.addWidget(self.view,1)
        nav=QHBoxLayout();
        for text, fn in (("Previous",lambda:self.navigate(-1)),("Next",lambda:self.navigate(1)),("Delete selected ROI(s)",self.delete_selected)):
            b=QPushButton(text); b.clicked.connect(fn); nav.addWidget(b)
        center.addLayout(nav); self.status=QLabel(); self.status.setAlignment(Qt.AlignmentFlag.AlignCenter); center.addWidget(self.status); layout.addLayout(center,4)
        self.delete_shortcut=QShortcut(QKeySequence(Qt.Key.Key_Delete),self); self.delete_shortcut.activated.connect(self.delete_selected)
        self.copy_shortcut=QShortcut(QKeySequence.StandardKey.Copy,self); self.copy_shortcut.activated.connect(self.copy_selected_rois)
        self.paste_shortcut=QShortcut(QKeySequence.StandardKey.Paste,self); self.paste_shortcut.activated.connect(self.paste_rois)
        self.previous_roi_shortcut=QShortcut(QKeySequence(Qt.Key.Key_Left),self.view); self.previous_roi_shortcut.activated.connect(lambda:self.navigate_roi(-1))
        self.next_roi_shortcut=QShortcut(QKeySequence(Qt.Key.Key_Right),self.view); self.next_roi_shortcut.activated.connect(lambda:self.navigate_roi(1))
        self.previous_image_shortcut=QShortcut(QKeySequence(Qt.Key.Key_Up),self.view); self.previous_image_shortcut.activated.connect(lambda:self.navigate(-1))
        self.next_image_shortcut=QShortcut(QKeySequence(Qt.Key.Key_Down),self.view); self.next_image_shortcut.activated.connect(lambda:self.navigate(1))
        for shortcut in (self.previous_roi_shortcut,self.next_roi_shortcut,self.previous_image_shortcut,self.next_image_shortcut):
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.list_previous_roi_shortcut=QShortcut(QKeySequence(Qt.Key.Key_Left),self.images); self.list_previous_roi_shortcut.activated.connect(lambda:self.navigate_roi(-1))
        self.list_next_roi_shortcut=QShortcut(QKeySequence(Qt.Key.Key_Right),self.images); self.list_next_roi_shortcut.activated.connect(lambda:self.navigate_roi(1))
        self.list_previous_roi_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.list_next_roi_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.next_class_shortcut=QShortcut(QKeySequence(Qt.Key.Key_Tab),self); self.next_class_shortcut.activated.connect(lambda:self.cycle_selected_classes(1))
        self.previous_class_shortcut=QShortcut(QKeySequence(Qt.Key.Key_Backtab),self); self.previous_class_shortcut.activated.connect(lambda:self.cycle_selected_classes(-1))
        right=QVBoxLayout()

        classes_section=QVBoxLayout(); classes_section.addWidget(QLabel("Ctrl-select ROIs; Tab / Shift+Tab changes class"))
        self.class_list=QListWidget(); classes_section.addWidget(self.class_list)
        b=QPushButton("Add class..."); b.clicked.connect(self.add_class); classes_section.addWidget(b); b=QPushButton("Assign class to selected ROI(s)"); b.clicked.connect(self.assign_class); classes_section.addWidget(b)
        for text, fn in (("Set all current ROI labels",lambda:self.set_all_labels(False)),("Set all image ROI labels",lambda:self.set_all_labels(True))):
            b=QPushButton(text); b.clicked.connect(fn); classes_section.addWidget(b)
        right.addWidget(CollapsibleSection("Classes",classes_section,True))

        patterns_section=QVBoxLayout(); self.pattern_list=QListWidget(); self.pattern_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection); patterns_section.addWidget(self.pattern_list)
        for text, fn in (("Draw / learn anchor pattern",self.begin_pattern),("Delete selected pattern(s)",self.delete_selected_patterns),("Extend current by median gap",lambda:self.extend_gap(False)),("Extend all by median gap",lambda:self.extend_gap(True)),("Auto-populate current",lambda:self.auto_place(False)),("Auto-populate all",lambda:self.auto_place(True))):
            b=QPushButton(text); b.clicked.connect(fn); patterns_section.addWidget(b)
        self.threshold=self.spin(.75,0,1,.05); patterns_section.addWidget(QLabel("Pattern threshold")); patterns_section.addWidget(self.threshold)
        self.occurrence_nms=self.spin(.30,0,1,.05); patterns_section.addWidget(QLabel("Occurrence NMS")); patterns_section.addWidget(self.occurrence_nms)
        patterns_section.addWidget(QLabel("Maximum occurrences / winning pattern")); self.maximum_occurrences=QSpinBox(); self.maximum_occurrences.setRange(1,10000); self.maximum_occurrences.setValue(1); patterns_section.addWidget(self.maximum_occurrences)
        right.addWidget(CollapsibleSection("Patterns and auto-population",patterns_section))

        roi_tools_section=QVBoxLayout()
        duplicate_row=QHBoxLayout(); duplicate_row.addWidget(QLabel("Duplicates / ROI")); self.duplicate_count=QSpinBox(); self.duplicate_count.setRange(1,1000); self.duplicate_count.setValue(10); duplicate_row.addWidget(self.duplicate_count); roi_tools_section.addLayout(duplicate_row)
        for text, fn in (("Duplicate current ROI(s)",lambda:self.duplicate_rois(False)),("Duplicate all ROI(s)",lambda:self.duplicate_rois(True))):
            b=QPushButton(text); b.clicked.connect(fn); roi_tools_section.addWidget(b)
        for text, fn in (("Expand current ROI(s)",lambda:self.random_resize(False)),("Expand all ROI(s)",lambda:self.random_resize(True))):
            b=QPushButton(text); b.clicked.connect(fn); roi_tools_section.addWidget(b)
        roi_tools_section.addWidget(QLabel("Expansion %: left / right / top / bottom (min-max)")); self.ranges=[]
        for side in ("Left","Right","Top","Bottom"):
            row=QHBoxLayout(); row.addWidget(QLabel(side)); lo=self.spin(0,0,100,1); hi=self.spin(0,0,100,1); row.addWidget(lo); row.addWidget(hi); roi_tools_section.addLayout(row); self.ranges.append((lo,hi))
        roi_tools_section.addWidget(QLabel("ROI width / height ratio (0 disables)")); self.ratio=self.spin(0,0,1000,.01); roi_tools_section.addWidget(self.ratio)
        right.addWidget(CollapsibleSection("ROI duplication and sizing",roi_tools_section))

        model_section=QVBoxLayout()
        b=QPushButton("Choose ONNX model..."); b.clicked.connect(self.choose_model); model_section.addWidget(b); self.model_label=QLabel(); self.model_label.setWordWrap(True); model_section.addWidget(self.model_label)
        for text, fn in (("Auto-label current",lambda:self.auto_label(False)),("Auto-label all",lambda:self.auto_label(True))): b=QPushButton(text); b.clicked.connect(fn); model_section.addWidget(b)
        right.addWidget(CollapsibleSection("ONNX auto-labeling",model_section))

        export_section=QVBoxLayout(); export_section.addWidget(QLabel("Split % (train / val / test)")); self.split=[self.spin(x,0,100,1) for x in (70,20,10)]; row=QHBoxLayout(); [row.addWidget(x) for x in self.split]; export_section.addLayout(row)
        self.group_duplicates=QCheckBox("Keep duplicate ROIs in one split")
        self.group_duplicates.setToolTip("When enabled, ROIs created together by Duplicate stay in one split even after resizing, moving, or relabeling.")
        export_section.addWidget(self.group_duplicates)
        self.group_by_source=QCheckBox("Keep all ROIs from same image in one split")
        self.group_by_source.setToolTip("When enabled, every ROI and duplicate from a source image stays together. This takes priority over duplicate grouping.")
        export_section.addWidget(self.group_by_source)
        self.balance=QCheckBox("Balance training split"); export_section.addWidget(self.balance); self.strategy=QListWidget(); self.strategy.addItems(["Oversample","Undersample"]); self.strategy.setCurrentRow(0); self.strategy.setMaximumHeight(45); export_section.addWidget(self.strategy)
        b=QPushButton("Export classification dataset..."); b.clicked.connect(self.export); export_section.addWidget(b)
        b=QPushButton("Export COCO detection dataset..."); b.clicked.connect(self.export_detection); export_section.addWidget(b)
        right.addWidget(CollapsibleSection("Dataset export",export_section,True))
        right.addStretch()
        right_panel=QWidget(); right_panel.setLayout(right)
        right_scroll=QScrollArea(); right_scroll.setWidgetResizable(True); right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); right_scroll.setWidget(right_panel); right_scroll.setMinimumWidth(300)
        layout.addWidget(right_scroll,1); self.update()
    @staticmethod
    def spin(value, low, high, step):
        b=QDoubleSpinBox(); b.setRange(low,high); b.setSingleStep(step); b.setDecimals(2 if step < 1 else 0); b.setValue(value); return b
    def export_progress(self, title, total):
        dialog=QProgressDialog("Preparing export...","Cancel",0,max(1,total),self)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setValue(0)
        def update_progress(value, maximum, message):
            dialog.setMaximum(max(1,maximum))
            dialog.setLabelText(message)
            dialog.setValue(value)
            QApplication.processEvents()
            if dialog.wasCanceled(): raise ExportCancelled()
        return dialog,update_progress
    def update(self, message=None):
        if message: self.status.setText(message); return
        n=len(self.entries[self.current].rois) if 0<=self.current<len(self.entries) else 0; self.status.setText(f"{self.current+1}/{len(self.entries)}  {n} ROI(s)  | Ctrl-select, right-drag move")
        self.model_label.setText(f"Model: {self.model_path}" if self.model_path else "Model: not selected")
    def add_images(self):
        files,_=QFileDialog.getOpenFileNames(self,"Add images","","Images (*.bmp *.jpg *.jpeg *.png *.tif *.tiff *.webp)"); self.add_paths([Path(x) for x in files])
    def add_folder(self):
        folder=QFileDialog.getExistingDirectory(self,"Add image folder"); self.add_paths(sorted(Path(folder).iterdir())) if folder else None
    def load_detection(self):
        folder=QFileDialog.getExistingDirectory(self,"Choose COCO detection dataset")
        if not folder: return
        try:
            loaded_entries,loaded_classes=load_coco_detection_dataset(folder)
        except Exception as e:
            QMessageBox.critical(self,"COCO load failed",str(e)); return
        if self.entries and QMessageBox.question(
            self,"Replace current dataset",
            "Loading COCO will replace the current images, ROIs, classes, and learned patterns. Continue?"
        ) != QMessageBox.StandardButton.Yes:
            return
        self.entries=loaded_entries
        self.classes=loaded_classes
        self.patterns=[]
        self.class_list.clear()
        self.class_list.addItems(self.classes)
        self.pattern_list.clear()
        self.images.clear()
        self.images.addItems([f"{len(entry.rois)} ROI | {entry.path.name}" for entry in self.entries])
        self.current=-1
        if self.entries: self.images.setCurrentRow(0)
        self.update(f"Loaded {len(self.entries)} image(s) and {sum(len(entry.rois) for entry in self.entries)} box(es)")
    def add_paths(self, paths):
        for path in paths:
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and not any(e.path.resolve()==path.resolve() for e in self.entries):
                try: image=load_image(path); self.entries.append(Entry(path.resolve(),image.size))
                except Exception: pass
        self.images.clear(); self.images.addItems([f"0 ROI | {e.path.name}" for e in self.entries]);
        if self.entries and self.current<0: self.images.setCurrentRow(0)
    def locate_image_by_hash(self):
        if not self.entries:
            self.update("Load images before locating by hash")
            return
        value,ok=QInputDialog.getText(
            self,"Locate image","12-character image hash:",
        )
        if not ok: return
        try:
            matches=find_entries_by_source_id(self.entries,value)
        except (OSError,ValueError) as e:
            QMessageBox.warning(self,"Invalid image hash",str(e))
            return
        if not matches:
            QMessageBox.information(self,"Image not found",f"No loaded image has hash {value.strip().lower()}.")
            return
        target=matches[0]
        self.images.clearSelection()
        self.images.setCurrentRow(target)
        self.images.item(target).setSelected(True)
        self.images.scrollToItem(self.images.item(target))
        self.images.setFocus()
        duplicate_note=f"; {len(matches)} identical-content images found" if len(matches)>1 else ""
        self.update(f"Located {self.entries[target].path.name} [{value.strip().lower()}]{duplicate_note}")
    def remove_selected_images(self):
        selected=sorted({index.row() for index in self.images.selectedIndexes()})
        if not selected:
            self.update("Select one or more images to remove")
            return
        old_current=self.current
        selected_set=set(selected)
        self.entries=[entry for index,entry in enumerate(self.entries) if index not in selected_set]
        self.images.blockSignals(True)
        self.images.clear()
        self.images.addItems([f"{len(entry.rois)} ROI | {entry.path.name}" for entry in self.entries])
        self.images.blockSignals(False)
        self.current=-1
        if self.entries:
            removed_before=sum(index<old_current for index in selected)
            next_row=max(0,min(len(self.entries)-1,old_current-removed_before))
            self.images.setCurrentRow(next_row)
        else:
            self.select(-1)
        self.update(f"Removed {len(selected)} image(s) from the builder; source files were not deleted")
    def remove_repeated_images(self):
        if not self.entries:
            self.update("No images to check")
            return
        unique=[]
        seen={}
        removed=[]
        failed=[]
        for entry in self.entries:
            try:
                digest=source_image_id(entry.path,40)
            except Exception:
                unique.append(entry); failed.append(entry.path.name); continue
            if digest in seen:
                removed.append((entry.path.name,seen[digest].path.name))
            else:
                seen[digest]=entry
                unique.append(entry)
        if not removed:
            message="No repeated images found."
            if failed: message+=f"\n\nCould not read {len(failed)} image(s)."
            QMessageBox.information(self,"Repeated images",message)
            return
        self.entries=unique
        self.images.clear()
        self.images.addItems([f"{len(e.rois)} ROI | {e.path.name}" for e in self.entries])
        self.current=-1
        if self.entries: self.images.setCurrentRow(0)
        message=f"Removed {len(removed)} repeated image(s); kept the first occurrence of each."
        if failed: message+=f"\n\nCould not read {len(failed)} image(s)."
        QMessageBox.information(self,"Repeated images",message)
    def select(self,index):
        self.current=index
        for item in self.items + self.pattern_items: self.scene.removeItem(item)
        self.items=[]; self.pattern_items=[]
        if not 0<=index<len(self.entries): self.pixmap.setPixmap(QPixmap()); self.update(); return
        self.entries[index].rois=sort_rois_reading_order(self.entries[index].rois)
        image=load_image(self.entries[index].path); self.pixmap.setPixmap(self.pixmap_from(image)); self.view.bounds=QRectF(0,0,*image.size); self.view.fitInView(self.view.bounds,Qt.AspectRatioMode.KeepAspectRatio)
        for matches in self.entries[index].pattern_matches.values():
            for x,y,w,h,score in matches:
                overlay=QGraphicsRectItem(QRectF(x,y,w,h)); overlay.setPen(QPen(Qt.GlobalColor.magenta,2,Qt.PenStyle.DashLine)); overlay.setBrush(QBrush(Qt.BrushStyle.NoBrush)); overlay.setZValue(1); overlay.setToolTip(f"Anchor match: {score:.3f}"); self.scene.addItem(overlay); self.pattern_items.append(overlay)
        for i,roi in enumerate(self.entries[index].rois): self.add_item(i,roi)
        self.update()
    @staticmethod
    def pixmap_from(image):
        image=image.convert("RGBA"); data=image.tobytes("raw","RGBA"); return QPixmap.fromImage(QImage(data,image.width,image.height,QImage.Format.Format_RGBA8888).copy())
    def add_item(self,index,roi):
        item=RoiItem(roi.rect,self.view.bounds,f"{index+1}:{self.classes[roi.class_id] if roi.class_id is not None and roi.class_id<len(self.classes) else '?'}",lambda r,i=index:self.changed(i,r),self.sort_current_rois,self.ratio.value); self.scene.addItem(item); self.items.append(item)
    def changed(self,index,r):
        if 0<=self.current<len(self.entries) and index<len(self.entries[self.current].rois): self.entries[self.current].rois[index].rect=(round(r.x()),round(r.y()),round(r.width()),round(r.height())); self.images.item(self.current).setText(f"{len(self.entries[self.current].rois)} ROI | {self.entries[self.current].path.name}")
    def sort_current_rois(self):
        if not 0<=self.current<len(self.entries): return
        entry=self.entries[self.current]
        selected_ids={id(entry.rois[index]) for index,item in enumerate(self.items) if index<len(entry.rois) and item.isSelected()}
        entry.rois=sort_rois_reading_order(entry.rois)
        self.select(self.current)
        for index,roi in enumerate(entry.rois):
            if id(roi) in selected_ids: self.items[index].setSelected(True)
    def created(self,r):
        if not 0<=self.current<len(self.entries): return
        rect=(round(r.x()),round(r.y()),round(r.width()),round(r.height())); ratio=self.ratio.value();
        if self.learning_pattern:
            self.learn_pattern(rect)
            return
        rect=force_ratio(rect,ratio,self.entries[self.current].size)
        if not any(iou(rect,x.rect)>.5 for x in self.entries[self.current].rois): self.entries[self.current].rois.append(ROI(rect)); self.select(self.current)
    def delete_selected(self):
        if not 0<=self.current<len(self.entries): return
        chosen={i for i,item in enumerate(self.items) if item.isSelected()}
        if not chosen:
            self.update("Select one or more ROI(s) before deleting")
            return
        self.entries[self.current].rois=[roi for i,roi in enumerate(self.entries[self.current].rois) if i not in chosen]
        self.select(self.current)
    def copy_selected_rois(self):
        if not 0<=self.current<len(self.entries): return
        selected=[i for i,item in enumerate(self.items) if item.isSelected()]
        if not selected:
            self.update("Select one or more ROI(s), then press Ctrl+C")
            return
        self.roi_clipboard=[]
        for index in selected:
            roi=self.entries[self.current].rois[index]
            if roi.duplicate_group is None:
                roi.duplicate_group=id(roi)
            self.roi_clipboard.append(ROI(roi.rect,roi.class_id,roi.confidence,roi.duplicate_group))
        self.update(f"Copied {len(self.roi_clipboard)} ROI(s)")
    def paste_rois(self):
        if not self.roi_clipboard:
            self.update("Copy one or more ROI(s) before pasting")
            return
        if not 0<=self.current<len(self.entries): return
        entry=self.entries[self.current]
        pasted=[]
        for roi in self.roi_clipboard:
            x,y,width,height=roi.rect
            width=min(width,entry.size[0]); height=min(height,entry.size[1])
            rect=force_ratio(
                (min(entry.size[0]-width,x+10),min(entry.size[1]-height,y+10),width,height),
                self.ratio.value(),entry.size
            )
            pasted_roi=ROI(rect,roi.class_id,roi.confidence,roi.duplicate_group)
            entry.rois.append(pasted_roi)
            pasted.append(pasted_roi)
        self.select(self.current)
        pasted_ids={id(roi) for roi in pasted}
        for index,roi in enumerate(entry.rois):
            if id(roi) in pasted_ids: self.items[index].setSelected(True)
        self.update(f"Pasted {len(self.roi_clipboard)} ROI(s)")
    def clear_current(self):
        if 0<=self.current<len(self.entries): self.entries[self.current].rois.clear(); self.select(self.current)
    def clear_all(self):
        for e in self.entries: e.rois.clear()
        self.select(self.current)
    def clear_images(self):
        self.entries.clear()
        self.images.clear()
        self.select(-1)
        self.update(f"Cleared images; retained {len(self.patterns)} learned pattern(s)")
    def navigate(self, amount):
        if self.entries: self.images.setCurrentRow(max(0,min(len(self.entries)-1,self.current+amount)))
    def navigate_roi(self, amount):
        if not 0<=self.current<len(self.entries) or not self.items:
            self.update("No ROI available on the current image")
            return
        selected=[index for index,item in enumerate(self.items) if item.isSelected()]
        if selected:
            anchor=max(selected) if amount>0 else min(selected)
            target=(anchor+amount)%len(self.items)
        else:
            target=0 if amount>0 else len(self.items)-1
        for item in self.items: item.setSelected(False)
        self.items[target].setSelected(True)
        self.view.ensureVisible(self.items[target])
        roi=self.entries[self.current].rois[target]
        class_name=self.classes[roi.class_id] if roi.class_id is not None and 0<=roi.class_id<len(self.classes) else "unassigned"
        self.update(f"ROI {target+1}/{len(self.items)}: {class_name}")
    def add_class(self):
        value,ok=QInputDialog.getText(self,"Add class","Class name:");
        if ok:
            try: name=valid_name(value,"Class name")
            except ValueError as e: QMessageBox.warning(self,"Invalid class",str(e)); return
            if name.casefold() not in {x.casefold() for x in self.classes}: self.classes.append(name); self.class_list.addItem(name)
    def assign_class(self):
        cid=self.class_list.currentRow()
        if cid<0 or not 0<=self.current<len(self.entries): return
        for item,roi in zip(self.items,self.entries[self.current].rois):
            if item.isSelected(): roi.class_id=cid
        self.select(self.current)
    def cycle_selected_classes(self, step):
        if not self.classes or not 0<=self.current<len(self.entries):
            self.update("Add at least one class before cycling ROI labels")
            return
        selected=[i for i,item in enumerate(self.items) if item.isSelected()]
        if not selected:
            self.update("Select one or more ROI(s), then press Tab to change class")
            return
        rois=self.entries[self.current].rois
        for index in selected:
            current=rois[index].class_id
            if current is None or not 0<=current<len(self.classes):
                rois[index].class_id=0 if step>0 else len(self.classes)-1
            else:
                rois[index].class_id=(current+step)%len(self.classes)
            rois[index].confidence=None
        self.select(self.current)
        for index in selected:
            if index<len(self.items): self.items[index].setSelected(True)
        assigned={rois[index].class_id for index in selected}
        if len(assigned)==1: self.class_list.setCurrentRow(next(iter(assigned)))
        self.update(f"Changed class for {len(selected)} selected ROI(s)")
    def set_all_labels(self, all_images):
        class_id=self.class_list.currentRow()
        if not 0<=class_id<len(self.classes):
            self.update("Select a class before assigning labels")
            return
        targets=self.entries if all_images else ([self.entries[self.current]] if 0<=self.current<len(self.entries) else [])
        changed=0
        for entry in targets:
            for roi in entry.rois:
                roi.class_id=class_id
                roi.confidence=None
                changed+=1
        self.select(self.current)
        self.update(f"Set {changed} ROI label(s) to '{self.classes[class_id]}'")
    def begin_pattern(self):
        if not 0<=self.current<len(self.entries):
            QMessageBox.information(self,"No image","Select a reference image first."); return
        if not self.entries[self.current].rois:
            QMessageBox.information(self,"ROIs required","Create the linked ROI(s) before drawing the anchor pattern."); return
        self.pattern_roi_indices=[i for i,item in enumerate(self.items) if item.isSelected()] or list(range(len(self.items)))
        self.learning_pattern=True
        self.update(f"Draw the anchor pattern; {len(self.pattern_roi_indices)} ROI(s) will follow it")
    def delete_selected_patterns(self):
        selected_rows=sorted({index.row() for index in self.pattern_list.selectedIndexes()})
        selected=[index for index in selected_rows if 0<=index<len(self.patterns)]
        if not selected_rows:
            self.update("Select one or more learned patterns to delete")
            return
        if not selected:
            self.pattern_list.clear()
            self.pattern_list.addItems([f"{pattern.name} ({len(pattern.rois)} linked ROI)" for pattern in self.patterns])
            self.update("Pattern list was stale and has been refreshed")
            return
        removed_names={self.patterns[index].name for index in selected}
        selected_set=set(selected)
        self.patterns=[pattern for index,pattern in enumerate(self.patterns) if index not in selected_set]
        for entry in self.entries:
            for name in removed_names: entry.pattern_matches.pop(name,None)
        self.pattern_list.clear()
        self.pattern_list.addItems([f"{pattern.name} ({len(pattern.rois)} linked ROI)" for pattern in self.patterns])
        self.select(self.current)
        self.update(f"Deleted {len(selected)} learned pattern(s)")

    def learn_pattern(self, pattern_rect):
        self.learning_pattern=False
        if not 0<=self.current<len(self.entries): return
        image=np.asarray(load_image(self.entries[self.current].path).convert("L")); x,y,w,h=pattern_rect; template=image[y:y+h,x:x+w].copy();
        if template.size==0 or template.std()<1e-6: QMessageBox.warning(self,"Invalid pattern","Anchor has no visible detail."); return
        value,ok=QInputDialog.getText(self,"Pattern name","Name:",text=f"pattern_{len(self.patterns)+1}");
        if not ok: return
        try: name=valid_name(value,"Pattern name")
        except ValueError as e: QMessageBox.warning(self,"Invalid pattern",str(e)); return
        if name.casefold() in {pattern.name.casefold() for pattern in self.patterns}:
            QMessageBox.warning(self,"Duplicate pattern name","Pattern names must be unique."); return
        linked=[ROI(self.entries[self.current].rois[i].rect,self.entries[self.current].rois[i].class_id,self.entries[self.current].rois[i].confidence) for i in self.pattern_roi_indices if i<len(self.entries[self.current].rois)]
        self.patterns.append(Pattern(name,self.entries[self.current].path,pattern_rect,template,linked)); self.pattern_list.addItem(f"{name} ({len(linked)} linked ROI)"); self.update(f"Learned {name}; auto-populate can now place linked ROI(s)")
    def extend_gap(self, all_images):
        if not 0<=self.current<len(self.entries): return
        source=self.entries[self.current]
        if len(source.rois)<2:
            QMessageBox.information(self,"Need two ROIs","Draw at least two ROIs on the current image first."); return
        # Median-gap placement is deliberately size-neutral.  Use the
        # explicit Expand buttons when randomized side expansion is wanted.
        ranges=[(0.0,0.0)] * 4; rng=random.Random()
        try: generated=populate_edges([r.rect for r in source.rois],source.size,ranges,rng)
        except ValueError as e: QMessageBox.warning(self,"Cannot extend ROIs",str(e)); return
        targets=self.entries if all_images else [source]; added=0; skipped=[]
        for entry in targets:
            if entry is source:
                candidates=generated
            elif len(entry.rois)>=2:
                # Recalculate from the target image's own ROI positions.  This
                # avoids copying a reference image's small translation error.
                try: candidates=populate_edges([roi.rect for roi in entry.rois],entry.size,ranges,rng)
                except ValueError: candidates=[]
            else:
                # If the target has no local spacing information, align the
                # reference sequence through the learned anchor pattern.
                candidates=[]
                pattern,matches=best_pattern_matches(
                    np.asarray(load_image(entry.path).convert("L")),self.patterns,
                    self.threshold.value(),self.occurrence_nms.value(),self.maximum_occurrences.value()
                )
                entry.pattern_matches.clear()
                if pattern:
                    template_height,template_width=pattern.template.shape
                    entry.pattern_matches[pattern.name]=[(x,y,template_width,template_height,score) for x,y,score in matches]
                    for match_x,match_y,_ in matches:
                        dx,dy=match_x-pattern.rect[0],match_y-pattern.rect[1]
                        candidates.extend((x+dx,y+dy,w,h) for x,y,w,h in generated)
                if not candidates: skipped.append(entry.path.name)
            for rect in candidates:
                rect=force_ratio(rect,self.ratio.value(),entry.size)
                if 0<=rect[0] and 0<=rect[1] and rect[0]+rect[2]<=entry.size[0] and rect[1]+rect[3]<=entry.size[1] and not any(iou(rect,old.rect)>.5 for old in entry.rois):
                    entry.rois.append(ROI(rect)); added+=1
        self.select(self.current); self.update(f"Added {added} median-gap ROI(s)")
        if skipped and all_images:
            self.update(f"Added {added} ROI(s); skipped {len(skipped)} image(s) without local ROIs or an anchor match")
    def random_resize(self, all_images):
        if not 0<=self.current<len(self.entries): return
        ranges=[(a.value(),b.value()) for a,b in self.ranges]; rng=random.Random()
        if all_images:
            targets=self.entries
        else:
            selected=any(item.isSelected() for item in self.items)
            targets=[self.entries[self.current]]
        changed=0
        for entry in targets:
            for index,roi in enumerate(entry.rois):
                if not all_images and selected and (index>=len(self.items) or not self.items[index].isSelected()): continue
                roi.rect=force_ratio(randomize_sides(roi.rect,entry.size,ranges,rng),self.ratio.value(),entry.size); changed+=1
        self.select(self.current); self.update(f"Expanded {changed} ROI(s) on {'all images' if all_images else 'the current image'}")
    def duplicate_rois(self, all_images):
        if not self.entries: return
        count=self.duplicate_count.value()
        if all_images:
            targets=[(entry,None) for entry in self.entries]
        elif 0<=self.current<len(self.entries):
            selected=any(item.isSelected() for item in self.items)
            targets=[(self.entries[self.current],selected)]
        else:
            return
        duplicated=0
        for entry,selected_only in targets:
            source=list(entry.rois)
            if selected_only:
                indices={i for i,item in enumerate(self.items) if item.isSelected()}
                source=[roi for i,roi in enumerate(source) if i in indices]
            for roi in source:
                if roi.duplicate_group is None:
                    roi.duplicate_group=id(roi)
                entry.rois.extend(ROI(roi.rect,roi.class_id,roi.confidence,roi.duplicate_group) for _ in range(count))
                duplicated+=count
        self.select(self.current); self.update(f"Duplicated {duplicated} ROI(s) at their original locations")
    def auto_place(self, all_images):
        if not self.patterns or not self.entries: return
        source=self.entries[self.current] if 0<=self.current<len(self.entries) else None; targets=self.entries if all_images else ([source] if source else []); rng=random.Random()
        for entry in targets:
            image=np.asarray(load_image(entry.path).convert("L")); proposals=[]
            pattern,matches=best_pattern_matches(
                image,self.patterns,self.threshold.value(),self.occurrence_nms.value(),
                self.maximum_occurrences.value()
            )
            entry.pattern_matches.clear()
            if pattern:
                template_height,template_width=pattern.template.shape
                entry.pattern_matches[pattern.name]=[(x,y,template_width,template_height,score) for x,y,score in matches]
                for match_x,match_y,score in matches:
                    dx,dy=match_x-pattern.rect[0],match_y-pattern.rect[1]
                    for linked in pattern.rois:
                        x,y,w,h=linked.rect; translated=(x+dx,y+dy,w,h)
                        if translated[0]<0 or translated[1]<0 or translated[0]+w>entry.size[0] or translated[1]+h>entry.size[1]: continue
                        resized=force_ratio(translated,self.ratio.value(),entry.size)
                        proposals.append((score,ROI(resized,linked.class_id,linked.confidence)))
            # Score-ordered suppression prevents duplicate linked ROIs from
            # overlapping one another or already annotated ROIs.
            accepted=[]
            for _,candidate in sorted(proposals,key=lambda item:item[0],reverse=True):
                if any(iou(candidate.rect,old.rect)>.5 for old in entry.rois): continue
                if any(iou(candidate.rect,old.rect)>.5 for old in accepted): continue
                accepted.append(candidate); entry.rois.append(candidate)
        self.select(self.current); self.update("Auto-population complete")
    def choose_model(self):
        value,_=QFileDialog.getOpenFileName(self,"Choose ONNX classifier","","ONNX model (*.onnx)");
        if value: self.model_path=Path(value).resolve(); self.session=None; self.update()
    def load_model(self):
        if not self.model_path or not self.model_path.is_file(): raise ValueError("Choose an ONNX model first.")
        if self.session: return
        import onnxruntime as ort
        self.session=ort.InferenceSession(str(self.model_path),providers=["CPUExecutionProvider"]); inputs=self.session.get_inputs()
        if len(inputs)!=1: raise ValueError("Expected one model input.")
        self.input_name=inputs[0].name; self.kind="bw8" if self.input_name=="images_bw8_uint8_nchw" else "c24" if self.input_name=="images_c24_uint8_nhwc_bgr" else None
        if not self.kind: raise ValueError(f"Unsupported embedded classifier input: {self.input_name}")
        raw=self.session.get_modelmeta().custom_metadata_map.get("names", "{}")
        try: parsed=ast.literal_eval(raw); self.names={int(k):str(v) for k,v in (parsed.items() if isinstance(parsed,dict) else enumerate(parsed))}
        except (SyntaxError,ValueError): self.names={}
        for name in self.names.values():
            if name.casefold() not in {value.casefold() for value in self.classes}:
                self.classes.append(name); self.class_list.addItem(name)
    def auto_label(self, all_images):
        targets=self.entries if all_images else ([self.entries[self.current]] if 0<=self.current<len(self.entries) else [])
        try: self.load_model()
        except Exception as e: QMessageBox.critical(self,"ONNX load failed",str(e)); return
        count=0
        for entry in targets:
            image=load_image(entry.path).convert("RGB")
            for roi in entry.rois:
                x,y,w,h=roi.rect; crop=image.crop((x,y,x+w,y+h)); tensor=np.asarray(crop.convert("L"),dtype=np.uint8)[None,None] if self.kind=="bw8" else np.asarray(crop,dtype=np.uint8)[...,::-1].copy()[None]
                scores=self.session.run(None,{self.input_name:tensor})[0][0]; index=int(np.argmax(scores)); roi.class_id=index if index<len(self.classes) else roi.class_id; roi.confidence=float(scores[index]); count+=1
        self.select(self.current); self.update(f"Auto-labeled {count} ROI(s)")
    def export_detection(self):
        labeled_entries=[
            entry for entry in self.entries
            if any(roi.class_id is not None and 0 <= roi.class_id < len(self.classes) for roi in entry.rois)
        ]
        if not labeled_entries:
            QMessageBox.information(self,"No labeled ROIs","Assign a class before exporting."); return
        ratios=tuple(x.value()/100 for x in self.split)
        if abs(sum(ratios)-1)>1e-6:
            QMessageBox.warning(self,"Invalid split","Split percentages must add to 100%."); return
        parent=QFileDialog.getExistingDirectory(self,"Dataset parent")
        if not parent: return
        name,ok=QInputDialog.getText(self,"Dataset folder","Name:",text="detection_dataset_coco")
        if not ok: return
        try: output=Path(parent)/valid_name(name,"Dataset folder")
        except ValueError as e: QMessageBox.warning(self,"Invalid folder",str(e)); return
        if output.exists(): QMessageBox.warning(self,"Destination exists",str(output)); return
        dialog,progress=self.export_progress("Export COCO detection dataset",len(labeled_entries))
        try:
            image_count,annotation_count=export_coco_detection_dataset(
                labeled_entries,self.classes,output,ratios,progress=progress
            )
        except ExportCancelled:
            dialog.close()
            if output.exists(): shutil.rmtree(output)
            self.update("COCO export cancelled")
            return
        except Exception as e:
            dialog.close()
            if output.exists(): shutil.rmtree(output)
            QMessageBox.critical(self,"Export failed",str(e)); return
        dialog.setValue(dialog.maximum())
        dialog.close()
        QMessageBox.information(
            self,"Dataset created",
            f"Created COCO detection dataset with {image_count} image(s) and "
            f"{annotation_count} box(es):\n{output}"
        )
    def export(self):
        samples=[(e,r,i) for e in self.entries for i,r in enumerate(e.rois) if r.class_id is not None and 0<=r.class_id<len(self.classes)]
        if not samples: QMessageBox.information(self,"No labeled ROIs","Assign a class before exporting."); return
        ratios=tuple(x.value()/100 for x in self.split)
        if abs(sum(ratios)-1)>1e-6: QMessageBox.warning(self,"Invalid split","Split percentages must add to 100%."); return
        unsplit=ratios==(1.0,0.0,0.0)
        parent=QFileDialog.getExistingDirectory(self,"Dataset parent");
        if not parent: return
        name,ok=QInputDialog.getText(self,"Dataset folder","Name:",text="classification_dataset");
        if not ok: return
        try: output=Path(parent)/valid_name(name,"Dataset folder")
        except ValueError as e: QMessageBox.warning(self,"Invalid folder",str(e)); return
        if output.exists(): QMessageBox.warning(self,"Destination exists",str(output)); return
        by_class={i:[] for i in range(len(self.classes))}; [by_class[r.class_id].append((e,r,i)) for e,r,i in samples]
        rng=random.Random(42)
        splits={"":list(samples)} if unsplit else split_samples(samples,ratios,rng,self.group_duplicates.isChecked(),self.group_by_source.isChecked())
        if self.balance.isChecked():
            balance_key="" if unsplit else "train"
            groups={cid:[x for x in splits[balance_key] if x[1].class_id==cid] for cid in by_class}; target=max((len(x) for x in groups.values()),default=0) if self.strategy.currentRow()==0 else min((len(x) for x in groups.values()),default=0); splits[balance_key]=[x for cid,g in groups.items() for x in (g+[rng.choice(g) for _ in range(target-len(g))] if self.strategy.currentRow()==0 and g else g[:target])]
        sample_paths=sorted({e.path.resolve() for e,_,_ in samples})
        crop_total=sum(len(group) for group in splits.values())
        dialog,progress=self.export_progress("Export classification dataset",len(sample_paths)+crop_total)
        image_ids={}
        progress_value=0
        try:
            for path in sample_paths:
                image_ids[path]=source_image_id(path)
                progress_value+=1
                progress(progress_value,len(sample_paths)+crop_total,f"Preparing: {path.name}")
        except ExportCancelled:
            dialog.close()
            self.update("Classification export cancelled")
            return
        group_indices={}
        for entry in self.entries:
            entry_groups={}
            for index,roi in enumerate(entry.rois):
                group_key=("duplicate",roi.duplicate_group) if roi.duplicate_group is not None else ("roi",index)
                if group_key not in entry_groups:
                    entry_groups[group_key]=len(entry_groups)
                group_indices[(entry.path.resolve(),index)]=entry_groups[group_key]
        occurrences=defaultdict(int)
        written=0
        try:
            samples_by_path=defaultdict(list)
            for split,group in splits.items():
                for cid in by_class: (output/split/self.classes[cid]).mkdir(parents=True,exist_ok=True)
                for sample in group: samples_by_path[sample[0].path.resolve()].append((split,*sample))
            for path,path_samples in samples_by_path.items():
                image=load_image(path)
                for split,e,r,i in path_samples:
                    image_id=image_ids[path]
                    group_index=group_indices[(path,i)]
                    occurrence_key=(image_id,i,group_index)
                    occurrence=occurrences[occurrence_key]
                    occurrences[occurrence_key]+=1
                    base=f"{image_id}_r{i:04d}_g{group_index:04d}_n{occurrence:03d}.png"
                    x,y,w,h=r.rect
                    image.crop((x,y,x+w,y+h)).save(
                        output/split/self.classes[r.class_id]/base,compress_level=1
                    )
                    written+=1
                    progress_value+=1
                    progress(progress_value,len(sample_paths)+crop_total,f"Exporting {split or 'dataset'}: {path.name}")
        except ExportCancelled:
            dialog.close()
            if output.exists(): shutil.rmtree(output)
            self.update("Classification export cancelled")
            return
        except Exception as e:
            dialog.close()
            if output.exists(): shutil.rmtree(output)
            QMessageBox.critical(self,"Export failed",f"Export stopped after {written} crop(s).\n\n{e}"); return
        dialog.setValue(dialog.maximum())
        dialog.close()
        QMessageBox.information(self,"Dataset created",f"Created {written} crop(s):\n{output}")


def main():
    parser=ArgumentParser(description=__doc__); parser.add_argument("--model",type=Path); args=parser.parse_args(); app=QApplication(sys.argv); window=MainWindow(args.model); window.show(); sys.exit(app.exec())


if __name__ == "__main__": main()
