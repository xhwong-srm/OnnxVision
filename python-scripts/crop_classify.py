"""Interactive ROI crop tool.

Install:  pip install PySide6 Pillow
Run:      python crop.py
Classify: python crop.py --model path/to/best.pt
"""

from __future__ import annotations

import os
import random
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

IMAGE_FILTER = "Images (*.bmp *.jpg *.jpeg *.png *.tif *.tiff *.webp)"
SUPPORTED_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
ROI_ASPECT_RATIO = 3 / 4  # width / height


def aspect_ratio_rect(
    anchor: QPointF,
    pointer: QPointF,
    bounds: QRectF,
    horizontal_direction: int | None = None,
    vertical_direction: int | None = None,
    aspect_ratio: float = ROI_ASPECT_RATIO,
) -> QRectF:
    """Build a bounds-limited ROI anchored at one corner with a 3:4 ratio."""
    dx = pointer.x() - anchor.x()
    dy = pointer.y() - anchor.y()
    horizontal_direction = horizontal_direction or (1 if dx >= 0 else -1)
    vertical_direction = vertical_direction or (1 if dy >= 0 else -1)

    width = abs(dx)
    height = abs(dy)
    if width / aspect_ratio >= height:
        height = width / aspect_ratio
    else:
        width = height * aspect_ratio

    max_width = bounds.right() - anchor.x() if horizontal_direction > 0 else anchor.x() - bounds.left()
    max_height = bounds.bottom() - anchor.y() if vertical_direction > 0 else anchor.y() - bounds.top()
    scale = min(1.0, max_width / width if width else 1.0, max_height / height if height else 1.0)
    opposite = QPointF(
        anchor.x() + horizontal_direction * width * scale,
        anchor.y() + vertical_direction * height * scale,
    )
    return QRectF(anchor, opposite).normalized()


def auto_populated_rois(
    rois: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
    size_variation_percent: float = 0.0,
    rng: random.Random | None = None,
) -> list[tuple[int, int, int, int]]:
    """Extend a left-to-right ROI sequence to both image boundaries."""
    if len(rois) < 2:
        raise ValueError("At least two ROIs are required to determine the spacing.")

    ordered = sorted(rois, key=lambda roi: roi[0] + roi[2] / 2)
    centers = [(x + width / 2, y + height / 2) for x, y, width, height in ordered]
    steps = [
        (right_x - left_x, right_y - left_y)
        for (left_x, left_y), (right_x, right_y) in zip(centers, centers[1:])
    ]
    step_x = sum(step[0] for step in steps) / len(steps)
    step_y = sum(step[1] for step in steps) / len(steps)
    if step_x < 1:
        raise ValueError("The ROIs need distinct horizontal positions.")

    variation = max(0.0, size_variation_percent) / 100.0
    randomizer = rng or random.Random()
    image_width, image_height = image_size

    def extend(edge_index: int, direction: int) -> list[tuple[int, int, int, int]]:
        edge_roi = ordered[edge_index]
        center_x, center_y = centers[edge_index]
        generated: list[tuple[int, int, int, int]] = []
        position = 1
        while True:
            target_x = center_x + direction * step_x * position
            target_y = center_y + direction * step_y * position
            if target_x < 0 or target_x > image_width or target_y < 0 or target_y > image_height:
                break
            scale = randomizer.uniform(1.0 - variation, 1.0 + variation)
            width = max(1, round(edge_roi[2] * scale))
            height = max(1, round(edge_roi[3] * scale))
            x = round(target_x - width / 2)
            y = round(target_y - height / 2)
            if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
                break
            generated.append((x, y, width, height))
            position += 1
        return generated

    left = extend(0, -1)
    right = extend(-1, 1)
    return list(reversed(left)) + ordered + right


@dataclass
class ImageEntry:
    path: Path
    size: tuple[int, int]
    rois: list[tuple[int, int, int, int]] | None = None
    pattern_rect: tuple[int, int, int, int] | None = None
    pattern_score: float | None = None

    def __post_init__(self):
        if self.rois is None:
            self.rois = []


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).copy()


def pixmap_from_pil(image: Image.Image) -> QPixmap:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimage)


class RoiItem(QGraphicsRectItem):
    """Movable rectangle with four resize handles."""

    HANDLE_SIZE = 10.0

    def __init__(self, rect: QRectF, bounds: QRectF, label: str, changed, aspect_ratio=ROI_ASPECT_RATIO):
        super().__init__(QRectF(0, 0, rect.width(), rect.height()))
        self.setPos(rect.topLeft())
        self.bounds = bounds
        self.changed = changed
        self.label = label
        self.aspect_ratio = aspect_ratio
        self.resize_corner: str | None = None
        self.press_scene_pos = QPointF()
        self.start_scene_rect = QRectF()
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(Qt.GlobalColor.red, 2))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setZValue(1)

    def scene_roi(self) -> QRectF:
        return QRectF(self.pos(), self.rect().size()).normalized()

    def _handles(self) -> dict[str, QRectF]:
        half = self.HANDLE_SIZE / 2
        r = self.rect()
        return {
            "tl": QRectF(r.left() - half, r.top() - half, self.HANDLE_SIZE, self.HANDLE_SIZE),
            "tr": QRectF(r.right() - half, r.top() - half, self.HANDLE_SIZE, self.HANDLE_SIZE),
            "bl": QRectF(r.left() - half, r.bottom() - half, self.HANDLE_SIZE, self.HANDLE_SIZE),
            "br": QRectF(r.right() - half, r.bottom() - half, self.HANDLE_SIZE, self.HANDLE_SIZE),
        }

    def boundingRect(self) -> QRectF:
        half = self.HANDLE_SIZE / 2 + 2
        return self.rect().adjusted(-half, -half, half, half)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.rect())
        for handle in self._handles().values():
            path.addRect(handle)
        return path

    def paint(self, painter: QPainter, option, widget=None):
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawRect(self.rect())
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.setBrush(QBrush(Qt.GlobalColor.red))
        for handle in self._handles().values():
            painter.drawRect(handle)
        painter.setPen(QPen(Qt.GlobalColor.yellow))
        painter.drawText(self.rect().adjusted(5, 3, -3, -3), self.label)

    def hoverMoveEvent(self, event):
        corner = next((name for name, rect in self._handles().items() if rect.contains(event.pos())), None)
        cursors = {
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
        }
        self.setCursor(cursors.get(corner, Qt.CursorShape.SizeAllCursor))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        self.resize_corner = next(
            (name for name, rect in self._handles().items() if rect.contains(event.pos())), None
        )
        self.press_scene_pos = event.scenePos()
        self.start_scene_rect = self.scene_roi()
        if self.resize_corner:
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.resize_corner:
            super().mouseMoveEvent(event)
            self._clamp_position()
            self.changed(self.scene_roi())
            return

        anchors = {
            "tl": self.start_scene_rect.bottomRight(),
            "tr": self.start_scene_rect.bottomLeft(),
            "bl": self.start_scene_rect.topRight(),
            "br": self.start_scene_rect.topLeft(),
        }
        directions = {
            "tl": (-1, -1),
            "tr": (1, -1),
            "bl": (-1, 1),
            "br": (1, 1),
        }
        horizontal_direction, vertical_direction = directions[self.resize_corner]
        r = aspect_ratio_rect(
            anchors[self.resize_corner],
            event.scenePos(),
            self.bounds,
            horizontal_direction,
            vertical_direction,
            self.aspect_ratio,
        )
        if r.width() >= 1 and r.height() >= 1:
            self.prepareGeometryChange()
            self.setPos(r.topLeft())
            self.setRect(0, 0, r.width(), r.height())
            self.changed(r)
        event.accept()

    def mouseReleaseEvent(self, event):
        was_resizing = self.resize_corner is not None
        self.resize_corner = None
        self.changed(self.scene_roi())
        if was_resizing:
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _clamp_position(self):
        r = self.scene_roi()
        x = min(max(r.left(), self.bounds.left()), self.bounds.right() - r.width())
        y = min(max(r.top(), self.bounds.top()), self.bounds.bottom() - r.height())
        self.setPos(x, y)


class ImageView(QGraphicsView):
    def __init__(self, roi_created):
        super().__init__()
        self.roi_created = roi_created
        self.image_bounds = QRectF()
        self.aspect_ratio = ROI_ASPECT_RATIO
        self.constrain_aspect_ratio = True
        self.drawing = False
        self.start = QPointF()
        self.draft: QGraphicsRectItem | None = None
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        item = self.itemAt(event.position().toPoint())
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.image_bounds.contains(scene_pos)
            and not isinstance(item, RoiItem)
        ):
            self.drawing = True
            self.start = scene_pos
            self.draft = QGraphicsRectItem()
            self.draft.setPen(QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.DashLine))
            self.scene().addItem(self.draft)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.draft:
            end = self._bounded(self.mapToScene(event.position().toPoint()))
            rect = QRectF(self.start, end).normalized()
            if self.constrain_aspect_ratio:
                rect = aspect_ratio_rect(
                    self.start,
                    end,
                    self.image_bounds,
                    aspect_ratio=self.aspect_ratio,
                )
            self.draft.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and self.draft:
            rect = self.draft.rect().intersected(self.image_bounds)
            self.scene().removeItem(self.draft)
            self.draft = None
            self.drawing = False
            if rect.width() >= 1 and rect.height() >= 1:
                self.roi_created(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def _bounded(self, point: QPointF) -> QPointF:
        return QPointF(
            min(max(point.x(), self.image_bounds.left()), self.image_bounds.right()),
            min(max(point.y(), self.image_bounds.top()), self.image_bounds.bottom()),
        )


class MainWindow(QMainWindow):
    def __init__(self, initial_model: Path | None = None):
        super().__init__()
        self.classification_mode = True
        self.setWindowTitle("ROI Crop and Classification Tool")
        self.resize(1200, 760)
        self.entries: list[ImageEntry] = []
        self.current_index = -1
        self.roi_items: list[RoiItem] = []
        self.pattern_item: QGraphicsRectItem | None = None
        self.pattern_template: np.ndarray | None = None
        self.pattern_reference_path: Path | None = None
        self.drawing_pattern = False
        self.results: dict[tuple[Path, int], tuple[str, float]] = {}
        self.model_path = initial_model
        self.model: YOLO | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = QVBoxLayout()
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.select_image)
        left.addWidget(self.image_list, 1)
        open_button = QPushButton("Open images...")
        open_button.clicked.connect(self.open_images)
        left.addWidget(open_button)
        open_folder_button = QPushButton("Open folder...")
        open_folder_button.clicked.connect(self.open_folder)
        left.addWidget(open_folder_button)
        clear_images_button = QPushButton("Clear all images")
        clear_images_button.clicked.connect(self.clear_all_images)
        left.addWidget(clear_images_button)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        self.scene = QGraphicsScene()
        self.view = ImageView(self.set_roi)
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        right.addWidget(self.view, 1)

        controls = QHBoxLayout()
        learn_pattern = QPushButton("Draw/Learn pattern")
        learn_pattern.clicked.connect(self.begin_pattern)
        controls.addWidget(learn_pattern)
        clear_pattern = QPushButton("Clear pattern")
        clear_pattern.clicked.connect(self.clear_pattern)
        controls.addWidget(clear_pattern)
        apply_all = QPushButton("Apply ROI to all")
        apply_all.clicked.connect(self.apply_roi_to_all)
        controls.addWidget(apply_all)
        populate_current = QPushButton("Auto-populate current")
        populate_current.clicked.connect(self.auto_populate_current)
        controls.addWidget(populate_current)
        populate_all = QPushButton("Auto-populate all")
        populate_all.clicked.connect(self.auto_populate_all)
        controls.addWidget(populate_all)
        clear_rois = QPushButton("Clear ROIs on current")
        clear_rois.clicked.connect(self.clear_rois)
        controls.addWidget(clear_rois)
        controls.addStretch()
        crop_current = QPushButton("Crop current...")
        crop_current.clicked.connect(lambda: self.choose_crop(False))
        controls.addWidget(crop_current)
        crop_all = QPushButton("Crop all...")
        crop_all.clicked.connect(lambda: self.choose_crop(True))
        controls.addWidget(crop_all)
        right.addLayout(controls)

        options = QHBoxLayout()
        options.addWidget(QLabel("ROI width / height:"))
        self.aspect_ratio = QDoubleSpinBox()
        self.aspect_ratio.setRange(0.25, 4.0)
        self.aspect_ratio.setDecimals(3)
        self.aspect_ratio.setSingleStep(0.05)
        self.aspect_ratio.setValue(ROI_ASPECT_RATIO)
        self.aspect_ratio.valueChanged.connect(self.set_aspect_ratio)
        options.addWidget(self.aspect_ratio)
        options.addWidget(QLabel("Pattern threshold:"))
        self.pattern_threshold = QDoubleSpinBox()
        self.pattern_threshold.setRange(0.0, 1.0)
        self.pattern_threshold.setDecimals(2)
        self.pattern_threshold.setSingleStep(0.05)
        self.pattern_threshold.setValue(0.75)
        options.addWidget(self.pattern_threshold)
        options.addWidget(QLabel("Auto size variation (+/- %):"))
        self.size_variation = QDoubleSpinBox()
        self.size_variation.setRange(0.0, 95.0)
        self.size_variation.setDecimals(1)
        self.size_variation.setSingleStep(1.0)
        self.size_variation.setValue(0.0)
        self.size_variation.setToolTip("Randomly scale each generated ROI while preserving its aspect ratio.")
        options.addWidget(self.size_variation)
        self.auto_number = QCheckBox("Auto-number cropped files (1_A, 1_B, ...)")
        self.auto_number.setChecked(True)
        options.addWidget(self.auto_number)
        options.addStretch()
        right.addLayout(options)

        model_controls = QHBoxLayout()
        choose_model = QPushButton("Choose trained model...")
        choose_model.clicked.connect(self.choose_model)
        model_controls.addWidget(choose_model)
        self.model_label = QLabel()
        self.model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        model_controls.addWidget(self.model_label, 1)
        self.classify_current_button = QPushButton("Classify current")
        self.classify_current_button.clicked.connect(lambda: self.classify(False))
        model_controls.addWidget(self.classify_current_button)
        self.classify_all_button = QPushButton("Classify all")
        self.classify_all_button.clicked.connect(lambda: self.classify(True))
        model_controls.addWidget(self.classify_all_button)
        right.addLayout(model_controls)

        self.status = QLabel(
            "Open images, draw ROIs, then choose a trained model or crop them."
        )
        right.addWidget(self.status)
        layout.addLayout(right, 4)
        self.update_model_label()

    def open_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open images", "", IMAGE_FILTER)
        self.add_paths([Path(path) for path in files])

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open image folder")
        if folder:
            paths = sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
            self.add_paths(paths)

    def add_paths(self, paths: list[Path]):
        known = {entry.path.resolve() for entry in self.entries}
        failures: list[str] = []
        for path in paths:
            if path.resolve() in known:
                continue
            try:
                with Image.open(path) as image:
                    size = ImageOps.exif_transpose(image).size
                self.entries.append(ImageEntry(path, size))
                self.image_list.addItem(path.name)
                known.add(path.resolve())
            except Exception as error:
                failures.append(f"{path.name}: {error}")
        if self.current_index < 0 and self.entries:
            self.image_list.setCurrentRow(0)
        if failures:
            QMessageBox.warning(self, "Some images could not be opened", "\n".join(failures))

    def select_image(self, index: int):
        if not 0 <= index < len(self.entries):
            return
        self.current_index = index
        try:
            image = load_image(self.entries[index].path)
        except Exception as error:
            QMessageBox.critical(self, "Open failed", str(error))
            return
        self.remove_roi_items()
        pixmap = pixmap_from_pil(image)
        self.pixmap_item.setPixmap(pixmap)
        self.view.image_bounds = QRectF(0, 0, pixmap.width(), pixmap.height())
        self.scene.setSceneRect(self.view.image_bounds)
        for roi_index, roi in enumerate(self.entries[index].rois or []):
            self.create_roi_item(QRectF(*roi), roi_index)
        self.show_pattern(self.entries[index])
        self.view.resetTransform()
        self.view.fitInView(self.view.image_bounds, Qt.AspectRatioMode.KeepAspectRatio)
        self.update_status()

    def set_roi(self, rect: QRectF):
        if self.current_index < 0:
            return
        if self.drawing_pattern:
            self.learn_pattern(rect)
            return
        roi_index = len(self.entries[self.current_index].rois or [])
        self.store_roi(roi_index, rect)
        self.create_roi_item(rect, roi_index)

    def create_roi_item(self, rect: QRectF, roi_index: int):
        item = RoiItem(
            rect,
            self.view.image_bounds,
            self.result_text(roi_index) if self.classification_mode else self.roi_label(roi_index),
            lambda changed_rect, index=roi_index: self.store_roi(index, changed_rect),
            self.view.aspect_ratio,
        )
        self.roi_items.append(item)
        self.scene.addItem(item)

    def set_aspect_ratio(self, value: float):
        self.view.aspect_ratio = value
        for item in self.roi_items:
            item.aspect_ratio = value

    def remove_roi_items(self):
        for item in self.roi_items:
            self.scene.removeItem(item)
        self.roi_items.clear()
        if self.pattern_item is not None:
            self.scene.removeItem(self.pattern_item)
            self.pattern_item = None

    def begin_pattern(self):
        if self.current_index < 0:
            QMessageBox.information(self, "No image", "Open and select a reference image first.")
            return
        self.drawing_pattern = True
        self.view.constrain_aspect_ratio = False
        self.status.setText("Draw a distinctive pattern on the reference image.")

    def learn_pattern(self, rect: QRectF):
        self.drawing_pattern = False
        self.view.constrain_aspect_ratio = True
        r = rect.normalized().intersected(self.view.image_bounds)
        x1, y1, x2, y2 = round(r.left()), round(r.top()), round(r.right()), round(r.bottom())
        if x2 <= x1 or y2 <= y1:
            return
        entry = self.entries[self.current_index]
        image = load_image(entry.path).convert("L")
        self.pattern_template = np.asarray(image.crop((x1, y1, x2, y2))).copy()
        self.pattern_reference_path = entry.path.resolve()
        entry.pattern_rect = (x1, y1, x2 - x1, y2 - y1)
        entry.pattern_score = 1.0
        self.show_pattern(entry)
        self.update_status("Pattern learned. Apply ROI to all will align each image before placing ROIs.")

    def clear_pattern(self):
        self.pattern_template = None
        self.pattern_reference_path = None
        self.drawing_pattern = False
        self.view.constrain_aspect_ratio = True
        for entry in self.entries:
            entry.pattern_rect = None
            entry.pattern_score = None
        if self.pattern_item is not None:
            self.scene.removeItem(self.pattern_item)
            self.pattern_item = None
        self.update_status("Pattern cleared; Apply ROI to all will copy absolute ROI positions.")

    def show_pattern(self, entry: ImageEntry):
        if self.pattern_item is not None:
            self.scene.removeItem(self.pattern_item)
            self.pattern_item = None
        if entry.pattern_rect is None:
            return
        self.pattern_item = QGraphicsRectItem(QRectF(*entry.pattern_rect))
        self.pattern_item.setPen(QPen(Qt.GlobalColor.cyan, 3))
        self.pattern_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.pattern_item.setZValue(2)
        self.scene.addItem(self.pattern_item)

    def store_roi(self, roi_index: int, rect: QRectF):
        if self.current_index < 0:
            return
        r = rect.normalized().intersected(self.view.image_bounds)
        x1, y1 = round(r.left()), round(r.top())
        x2, y2 = round(r.right()), round(r.bottom())
        if x2 > x1 and y2 > y1:
            rois = self.entries[self.current_index].rois
            assert rois is not None
            value = (x1, y1, x2 - x1, y2 - y1)
            old = rois[roi_index] if roi_index < len(rois) else None
            if roi_index == len(rois):
                rois.append(value)
            elif 0 <= roi_index < len(rois):
                rois[roi_index] = value
            if self.classification_mode and old != value:
                self.results.pop(self.result_key(self.entries[self.current_index], roi_index), None)
            if self.classification_mode and roi_index < len(self.roi_items):
                self.roi_items[roi_index].label = self.result_text(roi_index)
                self.roi_items[roi_index].update()
        self.update_status()

    def clear_rois(self):
        if self.current_index >= 0:
            if self.classification_mode:
                self.clear_results_for(self.entries[self.current_index])
            self.entries[self.current_index].rois = []
            self.remove_roi_items()
            self.update_status()

    def clear_all_images(self):
        if not self.entries:
            return
        answer = QMessageBox.question(self, "Clear all images?", "Remove every image from this batch?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.remove_roi_items()
        self.entries.clear()
        self.results.clear()
        self.pattern_template = None
        self.pattern_reference_path = None
        self.drawing_pattern = False
        self.view.constrain_aspect_ratio = True
        self.image_list.clear()
        self.current_index = -1
        self.pixmap_item.setPixmap(QPixmap())
        self.view.image_bounds = QRectF()
        self.scene.setSceneRect(QRectF())
        self.update_status()

    def auto_populate_current(self) -> bool:
        if self.current_index < 0:
            QMessageBox.information(self, "No image", "Open and select an image first.")
            return False
        rois = self.current_rois()
        if len(rois) < 2:
            QMessageBox.information(
                self,
                "Two ROIs required",
                "Draw at least two ROIs on the current image to determine their average spacing.",
            )
            return False
        try:
            populated = auto_populated_rois(
                rois,
                self.entries[self.current_index].size,
                self.size_variation.value(),
            )
        except ValueError as error:
            QMessageBox.warning(self, "Cannot auto-populate ROIs", str(error))
            return False
        added = len(populated) - len(rois)
        if self.classification_mode:
            self.clear_results_for(self.entries[self.current_index])
        self.entries[self.current_index].rois = populated
        self.select_image(self.current_index)
        self.update_status(f"Added {added} ROI(s) to the current image; {len(populated)} total.")
        return True

    def auto_populate_all(self):
        if self.current_index < 0:
            QMessageBox.information(self, "No image", "Open and select an image first.")
            return
        seed_rois = list(self.current_rois())
        if len(seed_rois) < 2:
            QMessageBox.information(
                self,
                "Two ROIs required",
                "Draw at least two ROIs on the current image to determine their average spacing.",
            )
            return
        if (
            self.pattern_reference_path is not None
            and self.entries[self.current_index].path.resolve() != self.pattern_reference_path
        ):
            QMessageBox.information(
                self,
                "Select pattern reference",
                "Select the image where the pattern was learned before auto-populating all images.",
            )
            return

        if self.pattern_template is not None and self.pattern_reference_path is not None:
            reference = self.entries[self.current_index]
            targets = self.apply_pattern_aligned_rois(reference, seed_rois)
        else:
            incompatible = [
                entry.path.name
                for entry in self.entries
                if any(x + width > entry.size[0] or y + height > entry.size[1] for x, y, width, height in seed_rois)
            ]
            if incompatible:
                QMessageBox.warning(
                    self,
                    "ROI does not fit all images",
                    "The ROI was not applied. It falls outside:\n" + "\n".join(incompatible[:12]),
                )
                return
            for entry in self.entries:
                if self.classification_mode:
                    self.clear_results_for(entry)
                entry.rois = list(seed_rois)
            targets = list(self.entries)

        populated = 0
        added = 0
        failures: list[str] = []
        for entry in targets:
            try:
                aligned_count = len(entry.rois or [])
                if self.classification_mode:
                    self.clear_results_for(entry)
                entry.rois = auto_populated_rois(
                    list(entry.rois or []),
                    entry.size,
                    self.size_variation.value(),
                )
                added += len(entry.rois) - aligned_count
                populated += 1
            except ValueError as error:
                failures.append(f"{entry.path.name}: {error}")

        self.select_image(self.current_index)
        message = f"Auto-populated {populated} image(s) to their boundaries; added {added} ROI(s)."
        self.update_status(message)
        if failures:
            QMessageBox.warning(self, "Some images could not be populated", message + "\n\n" + "\n".join(failures[:12]))

    def apply_roi_to_all(self):
        if self.pattern_template is not None and self.pattern_reference_path is not None:
            reference = next(
                (entry for entry in self.entries if entry.path.resolve() == self.pattern_reference_path),
                None,
            )
            if reference is None or not reference.rois:
                QMessageBox.information(
                    self,
                    "Reference ROI required",
                    "Draw at least one inspection ROI on the image where the pattern was learned.",
                )
                return
            self.apply_pattern_aligned_rois(reference, list(reference.rois))
            return
        rois = self.current_rois()
        if not rois:
            self.no_roi_message()
            return
        incompatible = [
            entry.path.name
            for entry in self.entries
            if any(x + width > entry.size[0] or y + height > entry.size[1] for x, y, width, height in rois)
        ]
        if incompatible:
            QMessageBox.warning(
                self,
                "ROI does not fit all images",
                "The ROI was not applied. It falls outside:\n" + "\n".join(incompatible[:12]),
            )
            return
        for entry in self.entries:
            if self.classification_mode:
                self.clear_results_for(entry)
            entry.rois = list(rois)
        self.update_status(f"Applied {len(rois)} ROI(s) to {len(self.entries)} images.")

    def apply_pattern_aligned_rois(
        self,
        reference: ImageEntry,
        rois: list[tuple[int, int, int, int]],
    ) -> list[ImageEntry]:
        if reference.pattern_rect is None or self.pattern_template is None:
            return []
        reference_x, reference_y, template_width, template_height = reference.pattern_rect
        threshold = self.pattern_threshold.value()
        failures: list[str] = []
        applied = 0
        skipped_rois = 0
        matched_entries: list[ImageEntry] = []

        for entry in self.entries:
            try:
                if self.classification_mode:
                    self.clear_results_for(entry)
                image = np.asarray(load_image(entry.path).convert("L"))
                if image.shape[0] < template_height or image.shape[1] < template_width:
                    raise ValueError("image is smaller than the learned pattern")
                result = cv2.matchTemplate(image, self.pattern_template, cv2.TM_CCOEFF_NORMED)
                _, score, _, match = cv2.minMaxLoc(result)
                if not np.isfinite(score) or score < threshold:
                    raise ValueError(f"match {score:.2f} is below threshold {threshold:.2f}")
                dx, dy = match[0] - reference_x, match[1] - reference_y
                translated = [(x + dx, y + dy, width, height) for x, y, width, height in rois]
                valid_rois = [
                    (x, y, width, height)
                    for x, y, width, height in translated
                    if x >= 0
                    and y >= 0
                    and x + width <= entry.size[0]
                    and y + height <= entry.size[1]
                ]
                skipped_rois += len(translated) - len(valid_rois)
                entry.rois = valid_rois
                entry.pattern_rect = (match[0], match[1], template_width, template_height)
                entry.pattern_score = score
                applied += 1
                matched_entries.append(entry)
            except Exception as error:
                failures.append(f"{entry.path.name}: {error}")

        self.select_image(self.current_index)
        message = f"Pattern-aligned ROI(s) on {applied} of {len(self.entries)} images."
        if skipped_rois:
            message += f" Skipped {skipped_rois} out-of-bounds ROI(s)."
        self.update_status(message)
        if failures:
            QMessageBox.warning(self, "Some pattern matches failed", message + "\n\n" + "\n".join(failures[:12]))
        return matched_entries

    def choose_model(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Choose trained classification model", "", "PyTorch model (*.pt)"
        )
        if filename:
            self.model_path = Path(filename).resolve()
            self.model = None
            self.results.clear()
            self.update_model_label()
            if self.current_index >= 0:
                self.select_image(self.current_index)

    def classify(self, all_images: bool):
        targets = self.entries if all_images else (
            [self.entries[self.current_index]] if self.current_index >= 0 else []
        )
        if not targets:
            QMessageBox.information(self, "No images", "Open at least one image first.")
            return
        if not self.model_path or not self.model_path.is_file():
            QMessageBox.information(self, "Model required", "Choose your trained best.pt model first.")
            self.choose_model()
            if not self.model_path or not self.model_path.is_file():
                return
        missing = [entry.path.name for entry in targets if not entry.rois]
        if missing:
            QMessageBox.warning(self, "Missing ROI", "Draw an ROI for:\n" + "\n".join(missing[:12]))
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.status.setText("Loading model and classifying ROI(s)...")
        QApplication.processEvents()
        try:
            if self.model is None:
                self.model = YOLO(str(self.model_path), task="classify")
            crops: list[Image.Image] = []
            keys: list[tuple[Path, int]] = []
            for entry in targets:
                image = load_image(entry.path).convert("RGB")
                for index, (x, y, width, height) in enumerate(entry.rois or []):
                    crops.append(image.crop((x, y, x + width, y + height)))
                    keys.append(self.result_key(entry, index))
            probabilities = self.predict_full_frame(crops)
            names = self.model.names
            for key, probs in zip(keys, probabilities):
                confidence, class_index = probs.max(dim=0)
                self.results[key] = (str(names[int(class_index)]), float(confidence))
            if self.current_index >= 0:
                self.select_image(self.current_index)
            self.update_status(f"Classified {len(crops)} ROI(s).")
        except Exception as error:
            QMessageBox.critical(self, "Classification failed", str(error))
            self.update_status()
        finally:
            QApplication.restoreOverrideCursor()

    def predict_full_frame(self, images: list[Image.Image]) -> torch.Tensor:
        """Use the same full-frame resize and scaling as the training dataset."""
        assert self.model is not None
        core = self.model.model
        imgsz_value = getattr(core, "args", {}).get("imgsz", 224)
        imgsz = int(imgsz_value[0] if isinstance(imgsz_value, (tuple, list)) else imgsz_value)
        preprocessing = transforms.Compose([
            transforms.Resize((imgsz, imgsz), antialias=True),
            transforms.ToTensor(),
        ])
        batch = torch.stack([preprocessing(image) for image in images])
        device = next(core.parameters()).device
        core.eval()
        with torch.inference_mode():
            output = core(batch.to(device))
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output.cpu()

    def result_text(self, roi_index: int) -> str:
        if not self.classification_mode or self.current_index < 0:
            return self.roi_label(roi_index)
        result = self.results.get(self.result_key(self.entries[self.current_index], roi_index))
        return self.roi_label(roi_index) if result is None else f"{result[0]}  {result[1]:.1%}"

    def result_key(self, entry: ImageEntry, roi_index: int) -> tuple[Path, int]:
        return entry.path.resolve(), roi_index

    def clear_results_for(self, entry: ImageEntry):
        path = entry.path.resolve()
        self.results = {key: value for key, value in self.results.items() if key[0] != path}

    def update_model_label(self):
        self.model_label.setText(
            f"Model: {self.model_path}" if self.model_path else "Model: not selected"
        )
        enabled = self.model_path is not None
        self.classify_current_button.setEnabled(enabled)
        self.classify_all_button.setEnabled(enabled)
        self.classify_current_button.setVisible(enabled)
        self.classify_all_button.setVisible(enabled)

    def choose_crop(self, all_images: bool):
        targets = self.entries if all_images else ([self.entries[self.current_index]] if self.current_index >= 0 else [])
        if not targets:
            QMessageBox.information(self, "No images", "Open at least one image first.")
            return
        missing = [entry.path.name for entry in targets if not entry.rois]
        if missing:
            QMessageBox.warning(self, "Missing ROI", "Set an ROI for:\n" + "\n".join(missing[:12]))
            return
        box = QMessageBox(self)
        box.setWindowTitle("Crop destination")
        box.setText(f"Where should the cropped {'images' if all_images else 'image'} be saved?")
        in_place = box.addButton("Crop in place", QMessageBox.ButtonRole.DestructiveRole)
        another_folder = box.addButton("Another folder...", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked == in_place:
            answer = QMessageBox.question(
                self,
                "Write crops beside originals?",
                "Single, non-numbered crops replace their originals. Multiple or auto-numbered crops are written beside them and may replace files with the same output names. Continue?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.crop_entries(targets, None, self.auto_number.isChecked())
        elif clicked == another_folder:
            folder = QFileDialog.getExistingDirectory(self, "Select output folder")
            if folder:
                self.crop_entries(targets, Path(folder), self.auto_number.isChecked())

    def crop_entries(self, entries: list[ImageEntry], output_folder: Path | None, auto_number: bool):
        failures: list[str] = []
        completed = 0
        for entry in entries:
            try:
                image_index = self.entries.index(entry) + 1
                image = load_image(entry.path)
                rois = entry.rois or []
                base_folder = output_folder or entry.path.parent
                for roi_index, (x, y, width, height) in enumerate(rois):
                    cropped = image.crop((x, y, x + width, y + height))
                    destination = self.crop_destination(
                        entry, base_folder, image_index, roi_index, len(rois), auto_number, output_folder is None
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temp = destination.with_name(f".{destination.stem}.crop-temp{destination.suffix}")
                    save_kwargs = {}
                    if image.info.get("icc_profile"):
                        save_kwargs["icc_profile"] = image.info["icc_profile"]
                    cropped.save(temp, **save_kwargs)
                    os.replace(temp, destination)
                    completed += 1
                if output_folder is None and len(rois) == 1 and not auto_number:
                    entry.size = (rois[0][2], rois[0][3])
                    entry.rois = [(0, 0, *entry.size)]
            except Exception as error:
                failures.append(f"{entry.path.name}: {error}")
        if output_folder is None and self.current_index >= 0:
            self.select_image(self.current_index)
        message = f"Created {completed} cropped file(s)."
        if output_folder:
            message += f"\nSaved to: {output_folder}"
        if failures:
            QMessageBox.warning(self, "Crop completed with errors", message + "\n\n" + "\n".join(failures))
        else:
            QMessageBox.information(self, "Crop complete", message)

    def crop_destination(
        self,
        entry: ImageEntry,
        folder: Path,
        image_index: int,
        roi_index: int,
        roi_count: int,
        auto_number: bool,
        in_place: bool,
    ) -> Path:
        suffix = entry.path.suffix
        roi_suffix = f"_{self.roi_label(roi_index)}" if roi_count > 1 else ""
        if auto_number:
            return folder / f"{image_index}{roi_suffix}{suffix}"
        if in_place and roi_count == 1:
            return entry.path
        return folder / f"{entry.path.stem}{roi_suffix}{suffix}"

    def current_rois(self) -> list[tuple[int, int, int, int]]:
        if self.current_index < 0:
            return []
        return self.entries[self.current_index].rois or []

    @staticmethod
    def roi_label(index: int) -> str:
        label = ""
        value = index + 1
        while value:
            value, remainder = divmod(value - 1, 26)
            label = chr(ord("A") + remainder) + label
        return label

    def no_roi_message(self):
        QMessageBox.information(self, "No ROI", "Drag on the current image to create an ROI first.")

    def update_status(self, message: str | None = None):
        if message:
            self.status.setText(message)
            return
        if self.current_index < 0:
            self.status.setText(
                "Open images, draw ROIs, then choose a trained model or crop them."
            )
            return
        rois = self.current_rois()
        if self.classification_mode:
            classified = sum(
                self.result_key(self.entries[self.current_index], i) in self.results
                for i in range(len(rois))
            )
            self.status.setText(
                f"{self.current_index + 1} / {len(self.entries)} — {self.entries[self.current_index].path.name} — "
                f"{len(rois)} ROI(s), {classified} classified"
            )
            return
        suffix = "No ROI — drag to create one." if not rois else f"{len(rois)} ROI(s): " + ", ".join(
            f"{self.roi_label(i)}={roi[2]}x{roi[3]} at ({roi[0]}, {roi[1]})" for i, roi in enumerate(rois)
        )
        self.status.setText(f"{self.current_index + 1} / {len(self.entries)} — {self.entries[self.current_index].path.name} — {suffix}")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Trained Ultralytics classification checkpoint (best.pt)")
    return parser.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args.model.resolve() if args.model else None)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
