"""Interactive ROI classification dataset builder.

The editor works on full source images.  ROIs can be drawn, moved with the
right mouse button, multi-selected with Ctrl, learned as a template pattern,
auto-populated to image edges, classified with an embedded ONNX classifier,
and exported as class-folder train/val/test crops.
"""
from __future__ import annotations

import ast
import hashlib
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
from PySide6.QtGui import QBrush, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QGraphicsItem,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QInputDialog, QLabel, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

SUPPORTED_SUFFIXES = {".bmp", ".dng", ".gif", ".heic", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
BAD = set('<>:"/\\|?*')
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


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


class RoiItem(QGraphicsRectItem):
    def __init__(self, rect, bounds, label, changed):
        super().__init__(QRectF(0, 0, rect[2], rect[3])); self.setPos(rect[0], rect[1]); self.bounds = bounds; self.label = label; self.changed = changed
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable); self.setAcceptHoverEvents(True); self.setZValue(2)
        self.setPen(QPen(Qt.GlobalColor.yellow, 2)); self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
    def scene_rect(self): return QRectF(self.pos(), self.rect().size())
    def boundingRect(self): return self.rect().adjusted(-5, -5, 5, 5)
    def paint(self, painter, option, widget=None):
        painter.setPen(QPen(Qt.GlobalColor.cyan if self.isSelected() else Qt.GlobalColor.yellow, 3)); painter.drawRect(self.rect()); painter.drawText(self.rect().adjusted(4, 2, -2, -2), self.label)
    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event); r = self.scene_rect(); self.setPos(max(0, min(r.x(), self.bounds.width() - r.width())), max(0, min(r.y(), self.bounds.height() - r.height()))); self.changed(self.scene_rect())
    def mouseReleaseEvent(self, event): self.changed(self.scene_rect()); super().mouseReleaseEvent(event)


class ImageView(QGraphicsView):
    def __init__(self, created):
        super().__init__(); self.created = created; self.bounds = QRectF(); self.drawing = False; self.start = QPointF(); self.draft = None; self.right_drag = False; self.last = QPointF(); self.setDragMode(QGraphicsView.DragMode.NoDrag)
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
        if self.right_drag: self.right_drag = False; event.accept(); return
        super().mouseReleaseEvent(event)
    def wheelEvent(self, event): self.scale(1.2 if event.angleDelta().y() > 0 else 1 / 1.2, 1.2 if event.angleDelta().y() > 0 else 1 / 1.2)


class MainWindow(QMainWindow):
    def __init__(self, initial_model=None):
        super().__init__(); self.setWindowTitle("Classification ROI Dataset Builder"); self.resize(1450, 850); self.entries=[]; self.classes=[]; self.patterns=[]; self.current=-1; self.items=[]; self.pattern_items=[]; self.learning_pattern=False; self.pattern_roi_indices=[]; self.model_path=Path(initial_model).resolve() if initial_model else None; self.session=None; self.input_name=None; self.kind=None; self.names={}
        root=QWidget(); self.setCentralWidget(root); layout=QHBoxLayout(root)
        left=QVBoxLayout(); left.addWidget(QLabel("Images")); self.images=QListWidget(); self.images.currentRowChanged.connect(self.select); left.addWidget(self.images,1)
        for text, fn in (("Add images...",self.add_images),("Add folder...",self.add_folder),("Remove repeated images",self.remove_repeated_images),("Delete ROIs on current",self.clear_current),("Delete ROIs on all",self.clear_all),("Clear images",self.clear_images)):
            b=QPushButton(text); b.clicked.connect(fn); left.addWidget(b)
        layout.addLayout(left,1)
        center=QVBoxLayout(); self.scene=QGraphicsScene(); self.pixmap=QGraphicsPixmapItem(); self.scene.addItem(self.pixmap); self.view=ImageView(self.created); self.view.setScene(self.scene); center.addWidget(self.view,1)
        nav=QHBoxLayout();
        for text, fn in (("Previous",lambda:self.navigate(-1)),("Next",lambda:self.navigate(1)),("Delete selected ROI(s)",self.delete_selected)):
            b=QPushButton(text); b.clicked.connect(fn); nav.addWidget(b)
        center.addLayout(nav); self.status=QLabel(); self.status.setAlignment(Qt.AlignmentFlag.AlignCenter); center.addWidget(self.status); layout.addLayout(center,4)
        self.delete_shortcut=QShortcut(QKeySequence(Qt.Key.Key_Delete),self); self.delete_shortcut.activated.connect(self.delete_selected)
        right=QVBoxLayout(); right.addWidget(QLabel("Classes (Ctrl-select ROIs; right-drag moves them)")); self.class_list=QListWidget(); right.addWidget(self.class_list,1)
        b=QPushButton("Add class..."); b.clicked.connect(self.add_class); right.addWidget(b); b=QPushButton("Assign class to selected ROI(s)"); b.clicked.connect(self.assign_class); right.addWidget(b)
        for text, fn in (("Set all current ROI labels",lambda:self.set_all_labels(False)),("Set all image ROI labels",lambda:self.set_all_labels(True))):
            b=QPushButton(text); b.clicked.connect(fn); right.addWidget(b)
        right.addWidget(QLabel("Learned anchor patterns")); self.pattern_list=QListWidget(); right.addWidget(self.pattern_list,1)
        for text, fn in (("Draw / learn anchor pattern",self.begin_pattern),("Extend current by median gap",lambda:self.extend_gap(False)),("Extend all by median gap",lambda:self.extend_gap(True)),("Auto-populate current",lambda:self.auto_place(False)),("Auto-populate all",lambda:self.auto_place(True)),("Expand current ROI(s)",lambda:self.random_resize(False)),("Expand all ROI(s)",lambda:self.random_resize(True))):
            b=QPushButton(text); b.clicked.connect(fn); right.addWidget(b)
        duplicate_row=QHBoxLayout(); duplicate_row.addWidget(QLabel("Duplicates / ROI")); self.duplicate_count=QSpinBox(); self.duplicate_count.setRange(1,1000); self.duplicate_count.setValue(10); duplicate_row.addWidget(self.duplicate_count); right.addLayout(duplicate_row)
        for text, fn in (("Duplicate current ROI(s)",lambda:self.duplicate_rois(False)),("Duplicate all ROI(s)",lambda:self.duplicate_rois(True))):
            b=QPushButton(text); b.clicked.connect(fn); right.addWidget(b)
        self.threshold=self.spin(.75,0,1,.05); right.addWidget(QLabel("Pattern threshold")); right.addWidget(self.threshold)
        self.occurrence_nms=self.spin(.30,0,1,.05); right.addWidget(QLabel("Occurrence NMS")); right.addWidget(self.occurrence_nms)
        right.addWidget(QLabel("Maximum occurrences / pattern")); self.maximum_occurrences=QSpinBox(); self.maximum_occurrences.setRange(1,10000); self.maximum_occurrences.setValue(500); right.addWidget(self.maximum_occurrences)
        right.addWidget(QLabel("Expansion %: left / right / top / bottom (min-max)")); self.ranges=[]
        for side in ("Left","Right","Top","Bottom"):
            row=QHBoxLayout(); row.addWidget(QLabel(side)); lo=self.spin(0,0,100,1); hi=self.spin(0,0,100,1); row.addWidget(lo); row.addWidget(hi); right.addLayout(row); self.ranges.append((lo,hi))
        right.addWidget(QLabel("ROI width / height ratio (0 disables)")); self.ratio=self.spin(0,0,1000,.01); right.addWidget(self.ratio)
        b=QPushButton("Choose ONNX model..."); b.clicked.connect(self.choose_model); right.addWidget(b); self.model_label=QLabel(); self.model_label.setWordWrap(True); right.addWidget(self.model_label)
        for text, fn in (("Auto-label current",lambda:self.auto_label(False)),("Auto-label all",lambda:self.auto_label(True))): b=QPushButton(text); b.clicked.connect(fn); right.addWidget(b)
        right.addWidget(QLabel("Split % (train / val / test)")); self.split=[self.spin(x,0,100,1) for x in (70,20,10)]; row=QHBoxLayout(); [row.addWidget(x) for x in self.split]; right.addLayout(row)
        self.group_duplicates=QCheckBox("Keep duplicate ROIs in one split")
        self.group_duplicates.setToolTip("When enabled, ROIs created together by Duplicate stay in one split even after resizing, moving, or relabeling.")
        right.addWidget(self.group_duplicates)
        self.group_by_source=QCheckBox("Keep all ROIs from same image in one split")
        self.group_by_source.setToolTip("When enabled, every ROI and duplicate from a source image stays together. This takes priority over duplicate grouping.")
        right.addWidget(self.group_by_source)
        self.balance=QCheckBox("Balance training split"); right.addWidget(self.balance); self.strategy=QListWidget(); self.strategy.addItems(["Oversample","Undersample"]); self.strategy.setCurrentRow(0); self.strategy.setMaximumHeight(45); right.addWidget(self.strategy)
        b=QPushButton("Export classification dataset..."); b.clicked.connect(self.export); right.addWidget(b)
        right_panel=QWidget(); right_panel.setLayout(right)
        right_scroll=QScrollArea(); right_scroll.setWidgetResizable(True); right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); right_scroll.setWidget(right_panel); right_scroll.setMinimumWidth(300)
        layout.addWidget(right_scroll,1); self.update()
    @staticmethod
    def spin(value, low, high, step):
        b=QDoubleSpinBox(); b.setRange(low,high); b.setSingleStep(step); b.setDecimals(2 if step < 1 else 0); b.setValue(value); return b
    def update(self, message=None):
        if message: self.status.setText(message); return
        n=len(self.entries[self.current].rois) if 0<=self.current<len(self.entries) else 0; self.status.setText(f"{self.current+1}/{len(self.entries)}  {n} ROI(s)  | Ctrl-select, right-drag move")
        self.model_label.setText(f"Model: {self.model_path}" if self.model_path else "Model: not selected")
    def add_images(self):
        files,_=QFileDialog.getOpenFileNames(self,"Add images","","Images (*.bmp *.jpg *.jpeg *.png *.tif *.tiff *.webp)"); self.add_paths([Path(x) for x in files])
    def add_folder(self):
        folder=QFileDialog.getExistingDirectory(self,"Add image folder"); self.add_paths(sorted(Path(folder).iterdir())) if folder else None
    def add_paths(self, paths):
        for path in paths:
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and not any(e.path.resolve()==path.resolve() for e in self.entries):
                try: image=load_image(path); self.entries.append(Entry(path.resolve(),image.size))
                except Exception: pass
        self.images.clear(); self.images.addItems([f"0 ROI | {e.path.name}" for e in self.entries]);
        if self.entries and self.current<0: self.images.setCurrentRow(0)
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
        item=RoiItem(roi.rect,self.view.bounds,f"{index+1}:{self.classes[roi.class_id] if roi.class_id is not None and roi.class_id<len(self.classes) else '?'}",lambda r,i=index:self.changed(i,r)); self.scene.addItem(item); self.items.append(item)
    def changed(self,index,r):
        if 0<=self.current<len(self.entries) and index<len(self.entries[self.current].rois): self.entries[self.current].rois[index].rect=(round(r.x()),round(r.y()),round(r.width()),round(r.height())); self.images.item(self.current).setText(f"{len(self.entries[self.current].rois)} ROI | {self.entries[self.current].path.name}")
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
    def clear_current(self):
        if 0<=self.current<len(self.entries): self.entries[self.current].rois.clear(); self.select(self.current)
    def clear_all(self):
        for e in self.entries: e.rois.clear()
        self.select(self.current)
    def clear_images(self): self.entries.clear(); self.patterns.clear(); self.images.clear(); self.select(-1)
    def navigate(self, amount):
        if self.entries: self.images.setCurrentRow(max(0,min(len(self.entries)-1,self.current+amount)))
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

    def learn_pattern(self, pattern_rect):
        self.learning_pattern=False
        if not 0<=self.current<len(self.entries): return
        image=np.asarray(load_image(self.entries[self.current].path).convert("L")); x,y,w,h=pattern_rect; template=image[y:y+h,x:x+w].copy();
        if template.size==0 or template.std()<1e-6: QMessageBox.warning(self,"Invalid pattern","Anchor has no visible detail."); return
        value,ok=QInputDialog.getText(self,"Pattern name","Name:",text=f"pattern_{len(self.patterns)+1}");
        if not ok: return
        try: name=valid_name(value,"Pattern name")
        except ValueError as e: QMessageBox.warning(self,"Invalid pattern",str(e)); return
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
                for pattern in self.patterns[:1]:
                    try: matches=find_pattern_matches(np.asarray(load_image(entry.path).convert("L")),pattern.template,self.threshold.value(),self.occurrence_nms.value(),self.maximum_occurrences.value())
                    except ValueError: continue
                    template_height,template_width=pattern.template.shape
                    entry.pattern_matches[pattern.name]=[(x,y,template_width,template_height,score) for x,y,score in matches]
                    for match_x,match_y,_ in matches:
                        dx,dy=match_x-pattern.rect[0],match_y-pattern.rect[1]
                        candidates.extend((x+dx,y+dy,w,h) for x,y,w,h in generated)
                    if candidates: break
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
            for pattern in self.patterns:
                try: matches=find_pattern_matches(image,pattern.template,self.threshold.value(),self.occurrence_nms.value(),self.maximum_occurrences.value())
                except ValueError: continue
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
    def export(self):
        samples=[(e,r,i) for e in self.entries for i,r in enumerate(e.rois) if r.class_id is not None and 0<=r.class_id<len(self.classes)]
        if not samples: QMessageBox.information(self,"No labeled ROIs","Assign a class before exporting."); return
        ratios=tuple(x.value()/100 for x in self.split)
        if abs(sum(ratios)-1)>1e-6: QMessageBox.warning(self,"Invalid split","Split percentages must add to 100%."); return
        parent=QFileDialog.getExistingDirectory(self,"Dataset parent");
        if not parent: return
        name,ok=QInputDialog.getText(self,"Dataset folder","Name:",text="classification_dataset");
        if not ok: return
        try: output=Path(parent)/valid_name(name,"Dataset folder")
        except ValueError as e: QMessageBox.warning(self,"Invalid folder",str(e)); return
        if output.exists(): QMessageBox.warning(self,"Destination exists",str(output)); return
        by_class={i:[] for i in range(len(self.classes))}; [by_class[r.class_id].append((e,r,i)) for e,r,i in samples]
        rng=random.Random(42)
        splits=split_samples(samples,ratios,rng,self.group_duplicates.isChecked(),self.group_by_source.isChecked())
        if self.balance.isChecked():
            groups={cid:[x for x in splits["train"] if x[1].class_id==cid] for cid in by_class}; target=max((len(x) for x in groups.values()),default=0) if self.strategy.currentRow()==0 else min((len(x) for x in groups.values()),default=0); splits["train"]=[x for cid,g in groups.items() for x in (g+[rng.choice(g) for _ in range(target-len(g))] if self.strategy.currentRow()==0 and g else g[:target])]
        image_ids={e.path.resolve():source_image_id(e.path) for e in self.entries}
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
            for split,group in splits.items():
                for cid in by_class: (output/split/self.classes[cid]).mkdir(parents=True,exist_ok=True)
                for e,r,i in group:
                    image_id=image_ids[e.path.resolve()]
                    group_index=group_indices[(e.path.resolve(),i)]
                    occurrence_key=(image_id,i,group_index)
                    occurrence=occurrences[occurrence_key]
                    occurrences[occurrence_key]+=1
                    base=f"{image_id}_r{i:04d}_g{group_index:04d}_n{occurrence:03d}.png"
                    image=load_image(e.path); x,y,w,h=r.rect; image.crop((x,y,x+w,y+h)).save(output/split/self.classes[r.class_id]/base); written+=1
        except Exception as e: QMessageBox.critical(self,"Export failed",f"{written} crop(s) written.\n\n{e}"); return
        QMessageBox.information(self,"Dataset created",f"Created {written} crop(s):\n{output}")


def main():
    parser=ArgumentParser(description=__doc__); parser.add_argument("--model",type=Path); args=parser.parse_args(); app=QApplication(sys.argv); window=MainWindow(args.model); window.show(); sys.exit(app.exec())


if __name__ == "__main__": main()
