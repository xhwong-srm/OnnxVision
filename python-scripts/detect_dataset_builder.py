"""Keyboard-first YOLO object-detection dataset builder.

Run:
    uv run python python-scripts/detect_dataset_builder.py

Draw boxes with the mouse, select a class with Tab, and move between images with
Left/Right. A learned pattern stores the current annotations relative to a
distinctive template. Any number of patterns can be learned and used to place
those annotations on the current image or the whole batch.
"""

from __future__ import annotations

import hashlib
import random
import shutil
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


IMAGE_FILTER = "Images (*.bmp *.jpg *.jpeg *.png *.tif *.tiff *.webp)"
SUPPORTED_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
INVALID_NAME_CHARACTERS = set('<>:"/\\|?*')
RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def validate_name(value: str, kind: str = "Name") -> str:
    name = value.strip()
    if not name or name in {".", ".."}:
        raise ValueError(f"{kind} cannot be empty.")
    if any(char in INVALID_NAME_CHARACTERS for char in name) or name.endswith((".", " ")):
        raise ValueError(f'{kind} cannot contain <>:"/\\|?* or end with a dot or space.')
    if name.split(".", 1)[0].upper() in RESERVED_NAMES:
        raise ValueError(f"'{name}' is a reserved Windows filename.")
    return name


def split_counts(count: int, ratios: tuple[float, float, float]) -> list[int]:
    exact = [count * ratio for ratio in ratios]
    result = [int(value) for value in exact]
    order = sorted(range(3), key=lambda index: exact[index] - result[index], reverse=True)
    for index in order[: count - sum(result)]:
        result[index] += 1
    return result


def yolo_line(class_id: int, rect: tuple[int, int, int, int], size: tuple[int, int]) -> str:
    x, y, width, height = rect
    image_width, image_height = size
    return (
        f"{class_id} {(x + width / 2) / image_width:.6f} "
        f"{(y + height / 2) / image_height:.6f} "
        f"{width / image_width:.6f} {height / image_height:.6f}"
    )


def intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def find_pattern_matches(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float,
    suppression_iou: float = 0.30,
    maximum_matches: int = 500,
) -> list[tuple[int, int, float]]:
    """Return distinct template matches ordered from highest to lowest score."""
    template_height, template_width = template.shape
    if image.shape[0] < template_height or image.shape[1] < template_width:
        raise ValueError("image is smaller than the pattern")
    scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(np.isfinite(scores) & (scores >= threshold))
    candidates = sorted(
        ((int(x), int(y), float(scores[y, x])) for x, y in zip(xs, ys)),
        key=lambda candidate: candidate[2],
        reverse=True,
    )
    selected: list[tuple[int, int, float]] = []
    for x, y, score in candidates:
        rect = (x, y, template_width, template_height)
        if any(
            intersection_over_union(
                rect,
                (kept_x, kept_y, template_width, template_height),
            )
            > suppression_iou
            for kept_x, kept_y, _kept_score in selected
        ):
            continue
        selected.append((x, y, score))
        if len(selected) >= maximum_matches:
            break
    return selected


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).copy()


def pixmap_from_pil(image: Image.Image) -> QPixmap:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimage)


@dataclass
class Annotation:
    class_id: int
    rect: tuple[int, int, int, int]


def suppress_overlapping_placements(
    candidates: list[tuple[float, Annotation]],
    iou_threshold: float = 0.50,
) -> list[Annotation]:
    """Apply score-ordered, class-aware NMS to boxes proposed by all patterns."""
    selected: list[Annotation] = []
    for _score, annotation in sorted(candidates, key=lambda candidate: candidate[0], reverse=True):
        if any(
            existing.class_id == annotation.class_id
            and intersection_over_union(existing.rect, annotation.rect) > iou_threshold
            for existing in selected
        ):
            continue
        selected.append(annotation)
    return selected


def randomize_box_sides(
    rect: tuple[int, int, int, int],
    image_size: tuple[int, int],
    minimum_percent: float,
    maximum_percent: float,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """Expand all four box sides independently, clamped to the image."""
    x, y, width, height = rect
    image_width, image_height = image_size
    low, high = sorted((max(0.0, minimum_percent), max(0.0, maximum_percent)))
    left = round(width * rng.uniform(low, high) / 100.0)
    right = round(width * rng.uniform(low, high) / 100.0)
    top = round(height * rng.uniform(low, high) / 100.0)
    bottom = round(height * rng.uniform(low, high) / 100.0)
    x1 = max(0, x - left)
    y1 = max(0, y - top)
    x2 = min(image_width, x + width + right)
    y2 = min(image_height, y + height + bottom)
    return x1, y1, x2 - x1, y2 - y1


@dataclass
class ImageEntry:
    path: Path
    size: tuple[int, int]
    annotations: list[Annotation] = field(default_factory=list)
    pattern_matches: dict[str, list[tuple[int, int, int, int, float]]] = field(
        default_factory=dict
    )


@dataclass
class LearnedPattern:
    name: str
    reference_path: Path
    reference_rect: tuple[int, int, int, int]
    template: np.ndarray
    annotations: list[Annotation]


class BoxItem(QGraphicsRectItem):
    HANDLE_SIZE = 10.0

    def __init__(self, rect: QRectF, bounds: QRectF, label: str, changed) -> None:
        super().__init__(QRectF(0, 0, rect.width(), rect.height()))
        self.setPos(rect.topLeft())
        self.bounds = bounds
        self.label = label
        self.changed = changed
        self.resize_corner: str | None = None
        self.start_rect = QRectF()
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(Qt.GlobalColor.red, 2))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setZValue(2)

    def scene_rect(self) -> QRectF:
        return QRectF(self.pos(), self.rect().size()).normalized()

    def handles(self) -> dict[str, QRectF]:
        half = self.HANDLE_SIZE / 2
        rect = self.rect()
        return {
            "tl": QRectF(rect.left() - half, rect.top() - half, self.HANDLE_SIZE, self.HANDLE_SIZE),
            "tr": QRectF(rect.right() - half, rect.top() - half, self.HANDLE_SIZE, self.HANDLE_SIZE),
            "bl": QRectF(rect.left() - half, rect.bottom() - half, self.HANDLE_SIZE, self.HANDLE_SIZE),
            "br": QRectF(rect.right() - half, rect.bottom() - half, self.HANDLE_SIZE, self.HANDLE_SIZE),
        }

    def boundingRect(self) -> QRectF:
        margin = self.HANDLE_SIZE / 2 + 2
        return self.rect().adjusted(-margin, -margin, margin, margin)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setPen(QPen(Qt.GlobalColor.yellow, 3) if self.isSelected() else self.pen())
        painter.setBrush(self.brush())
        painter.drawRect(self.rect())
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setBrush(QBrush(Qt.GlobalColor.red))
        for handle in self.handles().values():
            painter.drawRect(handle)
        painter.setPen(QPen(Qt.GlobalColor.yellow))
        painter.drawText(self.rect().adjusted(5, 3, -3, -3), self.label)

    def hoverMoveEvent(self, event) -> None:
        corner = next((name for name, rect in self.handles().items() if rect.contains(event.pos())), None)
        cursors = {
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
        }
        self.setCursor(cursors.get(corner, Qt.CursorShape.SizeAllCursor))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        self.resize_corner = next(
            (name for name, rect in self.handles().items() if rect.contains(event.pos())), None
        )
        self.start_rect = self.scene_rect()
        if self.resize_corner:
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if not self.resize_corner:
            super().mouseMoveEvent(event)
            rect = self.scene_rect()
            x = min(max(rect.left(), 0), self.bounds.width() - rect.width())
            y = min(max(rect.top(), 0), self.bounds.height() - rect.height())
            self.setPos(x, y)
            self.changed(self.scene_rect())
            return
        anchors = {
            "tl": self.start_rect.bottomRight(),
            "tr": self.start_rect.bottomLeft(),
            "bl": self.start_rect.topRight(),
            "br": self.start_rect.topLeft(),
        }
        rect = QRectF(anchors[self.resize_corner], event.scenePos()).normalized().intersected(self.bounds)
        if rect.width() >= 2 and rect.height() >= 2:
            self.prepareGeometryChange()
            self.setPos(rect.topLeft())
            self.setRect(0, 0, rect.width(), rect.height())
            self.changed(rect)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        resizing = self.resize_corner is not None
        self.resize_corner = None
        self.changed(self.scene_rect())
        if resizing:
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ImageView(QGraphicsView):
    def __init__(self, rectangle_created) -> None:
        super().__init__()
        self.rectangle_created = rectangle_created
        self.image_bounds = QRectF()
        self.drawing = False
        self.force_draw = False
        self.start = QPointF()
        self.draft: QGraphicsRectItem | None = None
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def mousePressEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        item = self.itemAt(event.position().toPoint())
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.image_bounds.contains(scene_pos)
            and (self.force_draw or not isinstance(item, BoxItem))
        ):
            self.drawing = True
            self.start = scene_pos
            self.draft = QGraphicsRectItem()
            self.draft.setPen(QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.DashLine))
            self.scene().addItem(self.draft)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.drawing and self.draft:
            point = self.mapToScene(event.position().toPoint())
            point = QPointF(
                min(max(point.x(), 0), self.image_bounds.width()),
                min(max(point.y(), 0), self.image_bounds.height()),
            )
            self.draft.setRect(QRectF(self.start, point).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.drawing and self.draft:
            rect = self.draft.rect().intersected(self.image_bounds)
            self.scene().removeItem(self.draft)
            self.draft = None
            self.drawing = False
            if rect.width() >= 2 and rect.height() >= 2:
                self.rectangle_created(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Object Detection Dataset Builder")
        self.resize(1450, 850)
        self.entries: list[ImageEntry] = []
        self.classes: list[str] = []
        self.patterns: list[LearnedPattern] = []
        self.current_index = -1
        self.box_items: list[BoxItem] = []
        self.pattern_items: list[QGraphicsRectItem] = []
        self.learning_pattern = False
        self.pattern_annotation_indices: list[int] = []

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = QVBoxLayout()
        left.addWidget(QLabel("Images"))
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.select_image)
        left.addWidget(self.image_list, 1)
        add_images = QPushButton("Add images...")
        add_images.clicked.connect(self.add_images)
        left.addWidget(add_images)
        add_folder = QPushButton("Add folder...")
        add_folder.clicked.connect(self.add_folder)
        left.addWidget(add_folder)
        clear_images = QPushButton("Clear images")
        clear_images.clicked.connect(self.clear_images)
        left.addWidget(clear_images)
        layout.addLayout(left, 1)

        center = QVBoxLayout()
        self.scene = QGraphicsScene()
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.view = ImageView(self.rectangle_created)
        self.view.setScene(self.scene)
        center.addWidget(self.view, 1)
        navigation = QHBoxLayout()
        previous = QPushButton("Previous (Left)")
        previous.clicked.connect(lambda: self.navigate(-1))
        navigation.addWidget(previous)
        delete = QPushButton("Delete selected box (Delete)")
        delete.clicked.connect(self.delete_selected_boxes)
        navigation.addWidget(delete)
        following = QPushButton("Next (Right)")
        following.clicked.connect(lambda: self.navigate(1))
        navigation.addWidget(following)
        center.addLayout(navigation)
        self.status = QLabel()
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(self.status)
        layout.addLayout(center, 4)

        right = QVBoxLayout()
        right.addWidget(QLabel("Classes (Tab cycles)"))
        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(lambda _row: self.update_status())
        right.addWidget(self.class_list, 1)
        add_class = QPushButton("Add class...")
        add_class.clicked.connect(lambda _checked=False: self.add_class())
        right.addWidget(add_class)
        remove_class = QPushButton("Remove class")
        remove_class.clicked.connect(self.remove_class)
        right.addWidget(remove_class)

        right.addWidget(QLabel("Learned patterns"))
        self.pattern_list = QListWidget()
        right.addWidget(self.pattern_list, 1)
        learn = QPushButton("Draw / learn another pattern")
        learn.clicked.connect(self.begin_pattern)
        right.addWidget(learn)
        remove_pattern = QPushButton("Remove selected pattern")
        remove_pattern.clicked.connect(self.remove_pattern)
        right.addWidget(remove_pattern)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Match threshold"))
        self.pattern_threshold = QDoubleSpinBox()
        self.pattern_threshold.setRange(0.0, 1.0)
        self.pattern_threshold.setDecimals(2)
        self.pattern_threshold.setSingleStep(0.05)
        self.pattern_threshold.setValue(0.75)
        threshold_row.addWidget(self.pattern_threshold)
        right.addLayout(threshold_row)
        self.auto_fit_boxes = QCheckBox("Auto-fit rough boxes using patterns")
        self.auto_fit_boxes.setChecked(True)
        self.auto_fit_boxes.setToolTip(
            "A rough box snaps to a learned placement of the selected class when a pattern match falls inside it."
        )
        right.addWidget(self.auto_fit_boxes)
        right.addWidget(QLabel("Auto-placement side expansion"))
        expansion_row = QHBoxLayout()
        self.expansion_min = self.percentage_box(0.0)
        self.expansion_max = self.percentage_box(0.0)
        for label, box in (("Min", self.expansion_min), ("Max", self.expansion_max)):
            column = QVBoxLayout()
            column.addWidget(QLabel(label))
            column.addWidget(box)
            expansion_row.addLayout(column)
        right.addLayout(expansion_row)
        expansion_help = QLabel("Left, right, top and bottom are randomized independently.")
        expansion_help.setWordWrap(True)
        expansion_help.setStyleSheet("color: #888888;")
        right.addWidget(expansion_help)
        auto_current = QPushButton("Auto-place boxes on current")
        auto_current.clicked.connect(lambda: self.auto_place(False))
        right.addWidget(auto_current)
        auto_all = QPushButton("Auto-place boxes on all")
        auto_all.clicked.connect(lambda: self.auto_place(True))
        right.addWidget(auto_all)

        right.addWidget(QLabel("Export split"))
        split_row = QHBoxLayout()
        self.train_ratio = self.ratio_box(70)
        self.val_ratio = self.ratio_box(20)
        self.test_ratio = self.ratio_box(10)
        for label, box in (("Train", self.train_ratio), ("Val", self.val_ratio), ("Test", self.test_ratio)):
            column = QVBoxLayout()
            column.addWidget(QLabel(label))
            column.addWidget(box)
            split_row.addLayout(column)
        right.addLayout(split_row)
        export = QPushButton("Generate YOLO dataset...")
        export.clicked.connect(self.export_dataset)
        right.addWidget(export)
        layout.addLayout(right, 1)

        self.add_shortcut(Qt.Key.Key_Left, lambda: self.navigate(-1))
        self.add_shortcut(Qt.Key.Key_Right, lambda: self.navigate(1))
        self.add_shortcut(Qt.Key.Key_Tab, lambda: self.cycle_class(1))
        self.add_shortcut(Qt.Key.Key_Backtab, lambda: self.cycle_class(-1))
        self.add_shortcut(Qt.Key.Key_Delete, self.delete_selected_boxes)
        self.update_status()

    @staticmethod
    def ratio_box(value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0, 100)
        box.setDecimals(0)
        box.setSuffix("%")
        box.setValue(value)
        return box

    @staticmethod
    def percentage_box(value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0, 100.0)
        box.setDecimals(1)
        box.setSingleStep(1.0)
        box.setSuffix("%")
        box.setValue(value)
        return box

    def add_shortcut(self, key: Qt.Key, action) -> None:
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut.activated.connect(action)

    def add_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add images", "", IMAGE_FILTER)
        self.add_paths([Path(filename) for filename in files])

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add image folder")
        if folder:
            self.add_paths(
                sorted(
                    path
                    for path in Path(folder).rglob("*")
                    if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )

    def add_paths(self, paths: list[Path]) -> None:
        known = {entry.path.resolve() for entry in self.entries}
        failures: list[str] = []
        for path in paths:
            if path.resolve() in known or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                image = load_image(path)
                self.entries.append(ImageEntry(path, image.size))
                self.image_list.addItem(path.name)
                known.add(path.resolve())
            except Exception as error:
                failures.append(f"{path.name}: {error}")
        if self.current_index < 0 and self.entries:
            self.image_list.setCurrentRow(0)
        self.refresh_image_list()
        if failures:
            QMessageBox.warning(self, "Some images were skipped", "\n".join(failures[:12]))

    def select_image(self, index: int) -> None:
        self.current_index = index
        self.remove_overlays()
        if not 0 <= index < len(self.entries):
            self.pixmap_item.setPixmap(QPixmap())
            self.view.image_bounds = QRectF()
            self.update_status()
            return
        entry = self.entries[index]
        pixmap = pixmap_from_pil(load_image(entry.path))
        self.pixmap_item.setPixmap(pixmap)
        self.view.image_bounds = QRectF(0, 0, pixmap.width(), pixmap.height())
        self.scene.setSceneRect(self.view.image_bounds)
        for annotation_index, annotation in enumerate(entry.annotations):
            self.add_box_item(annotation_index, annotation)
        for name, matches in entry.pattern_matches.items():
            for x, y, width, height, score in matches:
                item = QGraphicsRectItem(QRectF(x, y, width, height))
                item.setPen(QPen(Qt.GlobalColor.cyan, 2, Qt.PenStyle.DashLine))
                item.setToolTip(f"{name}: {score:.3f}")
                item.setZValue(1)
                self.scene.addItem(item)
                self.pattern_items.append(item)
        self.view.fitInView(self.view.image_bounds, Qt.AspectRatioMode.KeepAspectRatio)
        self.refresh_image_list()
        self.update_status()

    def remove_overlays(self) -> None:
        for item in self.box_items:
            self.scene.removeItem(item)
        for item in self.pattern_items:
            self.scene.removeItem(item)
        self.box_items.clear()
        self.pattern_items.clear()

    def rectangle_created(self, rect: QRectF) -> None:
        if self.learning_pattern:
            self.learn_pattern(rect)
            return
        if not 0 <= self.current_index < len(self.entries):
            return
        class_id = self.class_list.currentRow()
        if not 0 <= class_id < len(self.classes):
            QMessageBox.information(self, "Class required", "Add and select a class before drawing a box.")
            return
        manual_rect = self.rect_tuple(rect)
        fitted = self.best_fitted_annotation(manual_rect, class_id)
        annotation = fitted or Annotation(class_id, manual_rect)
        entry = self.entries[self.current_index]
        entry.annotations.append(annotation)
        self.add_box_item(len(entry.annotations) - 1, annotation)
        self.refresh_image_list()
        if fitted:
            self.update_status("Box auto-fitted to the nearest learned pattern placement.")
        else:
            self.update_status()

    def best_fitted_annotation(
        self,
        rough_rect: tuple[int, int, int, int],
        class_id: int,
    ) -> Annotation | None:
        if not self.auto_fit_boxes.isChecked() or not self.patterns:
            return None
        entry = self.entries[self.current_index]
        image = np.asarray(load_image(entry.path).convert("L"))
        rough_x, rough_y, rough_width, rough_height = rough_rect
        rough_center = (rough_x + rough_width / 2, rough_y + rough_height / 2)
        candidates: list[tuple[float, float, Annotation]] = []
        for pattern in self.patterns:
            try:
                matches = find_pattern_matches(
                    image,
                    pattern.template,
                    self.pattern_threshold.value(),
                )
            except ValueError:
                continue
            template_height, template_width = pattern.template.shape
            entry.pattern_matches[pattern.name] = [
                (x, y, template_width, template_height, score)
                for x, y, score in matches
            ]
            reference_x, reference_y, _, _ = pattern.reference_rect
            for match_x, match_y, score in matches:
                dx, dy = match_x - reference_x, match_y - reference_y
                for source in pattern.annotations:
                    if source.class_id != class_id:
                        continue
                    x, y, width, height = source.rect
                    translated = (x + dx, y + dy, width, height)
                    center_x, center_y = x + dx + width / 2, y + dy + height / 2
                    center_inside = (
                        rough_x <= center_x <= rough_x + rough_width
                        and rough_y <= center_y <= rough_y + rough_height
                    )
                    overlap = intersection_over_union(rough_rect, translated)
                    if not center_inside and overlap <= 0:
                        continue
                    distance = (center_x - rough_center[0]) ** 2 + (
                        center_y - rough_center[1]
                    ) ** 2
                    candidates.append((overlap, score - distance * 1e-9, Annotation(class_id, translated)))
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]

    @staticmethod
    def rect_tuple(rect: QRectF) -> tuple[int, int, int, int]:
        normalized = rect.normalized()
        x1, y1 = round(normalized.left()), round(normalized.top())
        x2, y2 = round(normalized.right()), round(normalized.bottom())
        return x1, y1, x2 - x1, y2 - y1

    def add_box_item(self, index: int, annotation: Annotation) -> None:
        label = self.classes[annotation.class_id] if annotation.class_id < len(self.classes) else str(annotation.class_id)
        item = BoxItem(
            QRectF(*annotation.rect),
            self.view.image_bounds,
            label,
            lambda rect, annotation_index=index: self.store_box(annotation_index, rect),
        )
        self.scene.addItem(item)
        self.box_items.append(item)

    def store_box(self, index: int, rect: QRectF) -> None:
        if 0 <= self.current_index < len(self.entries):
            annotations = self.entries[self.current_index].annotations
            if 0 <= index < len(annotations):
                annotations[index].rect = self.rect_tuple(rect.intersected(self.view.image_bounds))
                self.update_status()

    def delete_selected_boxes(self) -> None:
        if not 0 <= self.current_index < len(self.entries):
            return
        selected = [index for index, item in enumerate(self.box_items) if item.isSelected()]
        if not selected:
            return
        annotations = self.entries[self.current_index].annotations
        for index in reversed(selected):
            annotations.pop(index)
        self.select_image(self.current_index)

    def add_class(self) -> str | None:
        value, accepted = QInputDialog.getText(
            self, "Add class", "Class name:", QLineEdit.EchoMode.Normal, ""
        )
        if not accepted:
            return None
        try:
            name = validate_name(value, "Class name")
        except ValueError as error:
            QMessageBox.warning(self, "Invalid class name", str(error))
            return None
        existing = next((item for item in self.classes if item.casefold() == name.casefold()), None)
        if existing:
            self.class_list.setCurrentRow(self.classes.index(existing))
            return existing
        self.classes.append(name)
        self.class_list.addItem(name)
        self.class_list.setCurrentRow(len(self.classes) - 1)
        self.update_status()
        return name

    def remove_class(self) -> None:
        row = self.class_list.currentRow()
        if not 0 <= row < len(self.classes):
            return
        affected = sum(
            annotation.class_id == row
            for entry in self.entries
            for annotation in entry.annotations
        )
        prompt = f"Remove class '{self.classes[row]}'?"
        if affected:
            prompt += f"\n\nThis also removes {affected} box(es)."
        if QMessageBox.question(self, "Remove class?", prompt) != QMessageBox.StandardButton.Yes:
            return
        self.classes.pop(row)
        self.class_list.takeItem(row)
        for entry in self.entries:
            entry.annotations = [
                Annotation(
                    annotation.class_id - (annotation.class_id > row),
                    annotation.rect,
                )
                for annotation in entry.annotations
                if annotation.class_id != row
            ]
        for pattern in self.patterns:
            pattern.annotations = [
                Annotation(
                    annotation.class_id - (annotation.class_id > row),
                    annotation.rect,
                )
                for annotation in pattern.annotations
                if annotation.class_id != row
            ]
        self.select_image(self.current_index)

    def cycle_class(self, amount: int) -> None:
        if self.classes:
            self.class_list.setCurrentRow((self.class_list.currentRow() + amount) % len(self.classes))

    def begin_pattern(self) -> None:
        if not 0 <= self.current_index < len(self.entries):
            QMessageBox.information(self, "No image", "Add and select a reference image first.")
            return
        if not self.entries[self.current_index].annotations:
            QMessageBox.information(
                self,
                "Boxes required",
                "Draw the box(es) that this pattern should place before learning the pattern.",
            )
            return
        selected = [index for index, item in enumerate(self.box_items) if item.isSelected()]
        self.pattern_annotation_indices = selected or list(
            range(len(self.entries[self.current_index].annotations))
        )
        self.learning_pattern = True
        self.view.force_draw = True
        self.status.setText(
            f"Draw a tight alignment pattern; it will place {len(self.pattern_annotation_indices)} box(es)."
        )

    def learn_pattern(self, rect: QRectF) -> None:
        self.learning_pattern = False
        self.view.force_draw = False
        entry = self.entries[self.current_index]
        pattern_rect = self.rect_tuple(rect.intersected(self.view.image_bounds))
        x, y, width, height = pattern_rect
        image = np.asarray(load_image(entry.path).convert("L"))
        template = image[y : y + height, x : x + width].copy()
        if template.size == 0 or float(template.std()) < 1e-6:
            QMessageBox.warning(self, "Invalid pattern", "Choose a non-empty pattern with visible detail.")
            return
        suggested = f"pattern_{len(self.patterns) + 1}"
        value, accepted = QInputDialog.getText(
            self, "Pattern name", "Pattern name:", QLineEdit.EchoMode.Normal, suggested
        )
        if not accepted:
            self.update_status()
            return
        try:
            name = validate_name(value, "Pattern name")
        except ValueError as error:
            QMessageBox.warning(self, "Invalid pattern name", str(error))
            return
        if any(pattern.name.casefold() == name.casefold() for pattern in self.patterns):
            QMessageBox.warning(self, "Duplicate pattern", "Pattern names must be unique.")
            return
        self.patterns.append(
            LearnedPattern(
                name,
                entry.path.resolve(),
                pattern_rect,
                template,
                [
                    Annotation(
                        entry.annotations[index].class_id,
                        entry.annotations[index].rect,
                    )
                    for index in self.pattern_annotation_indices
                    if index < len(entry.annotations)
                ],
            )
        )
        self.pattern_list.addItem(f"{name} ({len(self.patterns[-1].annotations)} boxes)")
        entry.pattern_matches[name] = [(*pattern_rect, 1.0)]
        self.select_image(self.current_index)
        self.update_status(f"Learned '{name}'. You can learn more patterns or auto-place boxes.")

    def remove_pattern(self) -> None:
        row = self.pattern_list.currentRow()
        if not 0 <= row < len(self.patterns):
            return
        name = self.patterns.pop(row).name
        self.pattern_list.takeItem(row)
        for entry in self.entries:
            entry.pattern_matches.pop(name, None)
        self.select_image(self.current_index)

    def auto_place(self, all_images: bool) -> None:
        if not self.patterns:
            QMessageBox.information(self, "No patterns", "Learn at least one pattern first.")
            return
        targets = self.entries if all_images else (
            [self.entries[self.current_index]] if 0 <= self.current_index < len(self.entries) else []
        )
        threshold = self.pattern_threshold.value()
        expansion_min = self.expansion_min.value()
        expansion_max = self.expansion_max.value()
        if expansion_min > expansion_max:
            QMessageBox.warning(
                self,
                "Invalid expansion range",
                "Minimum side expansion cannot be greater than maximum side expansion.",
            )
            return
        randomizer = random.Random()
        placed = 0
        matches = 0
        failures: list[str] = []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for entry in targets:
                image = np.asarray(load_image(entry.path).convert("L"))
                proposed: list[tuple[float, Annotation]] = []
                for pattern in self.patterns:
                    template_height, template_width = pattern.template.shape
                    try:
                        found = find_pattern_matches(image, pattern.template, threshold)
                        if not found:
                            raise ValueError(f"no matches at or above {threshold:.2f}")
                        reference_x, reference_y, _, _ = pattern.reference_rect
                        entry.pattern_matches[pattern.name] = [
                            (x, y, template_width, template_height, score)
                            for x, y, score in found
                        ]
                        matches += len(found)
                        for match_x, match_y, score in found:
                            dx, dy = match_x - reference_x, match_y - reference_y
                            for source in pattern.annotations:
                                x, y, width, height = source.rect
                                translated = (x + dx, y + dy, width, height)
                                if (
                                    translated[0] < 0
                                    or translated[1] < 0
                                    or translated[0] + width > entry.size[0]
                                    or translated[1] + height > entry.size[1]
                                ):
                                    continue
                                proposed.append(
                                    (
                                        score,
                                        Annotation(
                                            source.class_id,
                                            randomize_box_sides(
                                                translated,
                                                entry.size,
                                                expansion_min,
                                                expansion_max,
                                                randomizer,
                                            ),
                                        ),
                                    )
                                )
                    except Exception as error:
                        failures.append(f"{entry.path.name} / {pattern.name}: {error}")
                for annotation in suppress_overlapping_placements(proposed):
                    overlaps_existing = any(
                        existing.class_id == annotation.class_id
                        and intersection_over_union(existing.rect, annotation.rect) > 0.50
                        for existing in entry.annotations
                    )
                    if not overlaps_existing:
                        entry.annotations.append(annotation)
                        placed += 1
        finally:
            QApplication.restoreOverrideCursor()
        self.select_image(self.current_index)
        message = f"Matched {matches} pattern(s) and placed {placed} new box(es)."
        self.update_status(message)
        if failures:
            QMessageBox.warning(
                self,
                "Some pattern matches failed",
                message + "\n\n" + "\n".join(failures[:12]),
            )

    def navigate(self, amount: int) -> None:
        if self.entries:
            current = max(self.current_index, 0)
            self.image_list.setCurrentRow(min(max(current + amount, 0), len(self.entries) - 1))

    def clear_images(self) -> None:
        if self.entries and QMessageBox.question(
            self, "Clear images?", "Remove all images, boxes, and learned patterns?"
        ) != QMessageBox.StandardButton.Yes:
            return
        self.remove_overlays()
        self.entries.clear()
        self.patterns.clear()
        self.pattern_annotation_indices.clear()
        self.learning_pattern = False
        self.view.force_draw = False
        self.image_list.clear()
        self.pattern_list.clear()
        self.current_index = -1
        self.pixmap_item.setPixmap(QPixmap())
        self.view.image_bounds = QRectF()
        self.update_status()

    def refresh_image_list(self) -> None:
        for index, entry in enumerate(self.entries):
            self.image_list.item(index).setText(
                f"{len(entry.annotations)} box(es)  |  {entry.path.name}"
            )
            self.image_list.item(index).setBackground(
                Qt.GlobalColor.darkGreen if entry.annotations else Qt.GlobalColor.transparent
            )

    def export_dataset(self) -> None:
        if not self.entries:
            QMessageBox.information(self, "No images", "Add images before exporting.")
            return
        if not self.classes:
            QMessageBox.information(self, "No classes", "Add at least one class before exporting.")
            return
        values = (self.train_ratio.value(), self.val_ratio.value(), self.test_ratio.value())
        if abs(sum(values) - 100.0) > 0.01:
            QMessageBox.warning(self, "Invalid split", "Train, val, and test must add up to 100%.")
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose the parent folder for the dataset")
        if not parent:
            return
        value, accepted = QInputDialog.getText(
            self,
            "Dataset folder",
            "New dataset folder name:",
            QLineEdit.EchoMode.Normal,
            "detection_dataset",
        )
        if not accepted:
            return
        try:
            dataset_name = validate_name(value, "Dataset folder name")
        except ValueError as error:
            QMessageBox.warning(self, "Invalid folder name", str(error))
            return
        output = Path(parent) / dataset_name
        if output.exists():
            QMessageBox.warning(self, "Destination exists", f"Nothing was changed:\n{output}")
            return

        entries = list(self.entries)
        random.Random(42).shuffle(entries)
        ratios = tuple(value / 100.0 for value in values)
        counts = split_counts(len(entries), ratios)  # type: ignore[arg-type]
        splits: dict[str, list[ImageEntry]] = {}
        offset = 0
        for split, count in zip(("train", "val", "test"), counts):
            splits[split] = entries[offset : offset + count]
            offset += count

        copied = 0
        try:
            for split, split_entries in splits.items():
                image_dir = output / "images" / split
                label_dir = output / "labels" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                used_names: set[str] = set()
                for entry in split_entries:
                    filename = entry.path.name
                    if filename.casefold() in used_names:
                        digest = hashlib.sha1(str(entry.path.resolve()).encode()).hexdigest()[:10]
                        filename = f"{entry.path.stem}_{digest}{entry.path.suffix.lower()}"
                    used_names.add(filename.casefold())
                    shutil.copy2(entry.path, image_dir / filename)
                    lines = [
                        yolo_line(annotation.class_id, annotation.rect, entry.size)
                        for annotation in entry.annotations
                    ]
                    (label_dir / f"{Path(filename).stem}.txt").write_text(
                        "\n".join(lines) + ("\n" if lines else ""),
                        encoding="utf-8",
                    )
                    copied += 1
            names = "\n".join(
                f"  {index}: {self.yaml_string(name)}" for index, name in enumerate(self.classes)
            )
            data_yaml = (
                "path: .\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
                "names:\n"
                f"{names}\n"
            )
            (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
        except Exception as error:
            QMessageBox.critical(
                self,
                "Export failed",
                f"Exported {copied} image(s) before the error.\n\n{error}\n\nPartial output: {output}",
            )
            return
        QMessageBox.information(
            self,
            "Dataset created",
            f"Created a YOLO detection dataset with {copied} image(s):\n{output}",
        )
        self.update_status(f"Dataset created: {output}")

    @staticmethod
    def yaml_string(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def update_status(self, message: str | None = None) -> None:
        if message:
            self.status.setText(message)
            return
        position = (
            f"{self.current_index + 1} / {len(self.entries)}"
            if self.current_index >= 0
            else f"0 / {len(self.entries)}"
        )
        boxes = (
            len(self.entries[self.current_index].annotations)
            if 0 <= self.current_index < len(self.entries)
            else 0
        )
        selected = (
            self.classes[self.class_list.currentRow()]
            if 0 <= self.class_list.currentRow() < len(self.classes)
            else "none"
        )
        mode = "DRAW PATTERN" if self.learning_pattern else "draw boxes"
        self.status.setText(
            f"{position}  •  {boxes} box(es)  •  class: {selected}  •  {mode}  •  Tab class, arrows navigate"
        )


def parse_args():
    return ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
