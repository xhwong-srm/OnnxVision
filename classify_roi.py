"""Interactive multi-ROI classification tool.

Run with the trained checkpoint:
    uv run python classify_roi.py --model runs/classify/yolo26-seal/weights/best.pt
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

from crop import (
    IMAGE_FILTER,
    ImageEntry,
    ImageView,
    RoiItem,
    load_image,
    pixmap_from_pil,
)
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QListWidget


class ClassificationWindow(QMainWindow):
    def __init__(self, initial_model: Path | None = None):
        super().__init__()
        self.setWindowTitle("ROI Classification Tool")
        self.resize(1250, 780)
        self.entries: list[ImageEntry] = []
        self.current_index = -1
        self.roi_items: list[RoiItem] = []
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
        open_images = QPushButton("Open images...")
        open_images.clicked.connect(self.open_images)
        left.addWidget(open_images)
        open_folder = QPushButton("Open folder...")
        open_folder.clicked.connect(self.open_folder)
        left.addWidget(open_folder)
        clear_images = QPushButton("Clear all images")
        clear_images.clicked.connect(self.clear_all_images)
        left.addWidget(clear_images)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        self.scene = QGraphicsScene()
        self.view = ImageView(self.set_roi)
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        right.addWidget(self.view, 1)

        roi_controls = QHBoxLayout()
        roi_controls.addWidget(QLabel("ROI width / height:"))
        self.aspect_ratio = QDoubleSpinBox()
        self.aspect_ratio.setRange(0.25, 4.0)
        self.aspect_ratio.setDecimals(3)
        self.aspect_ratio.setSingleStep(0.05)
        self.aspect_ratio.setValue(0.75)
        self.aspect_ratio.valueChanged.connect(self.set_aspect_ratio)
        roi_controls.addWidget(self.aspect_ratio)
        apply_all = QPushButton("Apply ROIs to all")
        apply_all.clicked.connect(self.apply_roi_to_all)
        roi_controls.addWidget(apply_all)
        clear_rois = QPushButton("Clear ROIs on current")
        clear_rois.clicked.connect(self.clear_rois)
        roi_controls.addWidget(clear_rois)
        roi_controls.addStretch()
        right.addLayout(roi_controls)

        model_controls = QHBoxLayout()
        choose_model = QPushButton("Choose trained model...")
        choose_model.clicked.connect(self.choose_model)
        model_controls.addWidget(choose_model)
        self.model_label = QLabel()
        self.model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        model_controls.addWidget(self.model_label, 1)
        classify_current = QPushButton("Classify current")
        classify_current.clicked.connect(lambda: self.classify(False))
        model_controls.addWidget(classify_current)
        classify_all = QPushButton("Classify all")
        classify_all.clicked.connect(lambda: self.classify(True))
        model_controls.addWidget(classify_all)
        right.addLayout(model_controls)

        self.status = QLabel("Open images, draw one or more ROIs, then choose a trained model.")
        right.addWidget(self.status)
        layout.addLayout(right, 4)
        self.update_model_label()

    def open_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open images", "", IMAGE_FILTER)
        self.add_paths([Path(path) for path in files])

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open image folder")
        if folder:
            from crop import SUPPORTED_SUFFIXES

            paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
            self.add_paths(paths)

    def add_paths(self, paths: list[Path]):
        known = {entry.path.resolve() for entry in self.entries}
        failures = []
        for path in paths:
            if path.resolve() in known:
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
        self.view.resetTransform()
        self.view.fitInView(self.view.image_bounds, Qt.AspectRatioMode.KeepAspectRatio)
        self.update_status()

    def set_roi(self, rect: QRectF):
        if self.current_index < 0:
            return
        index = len(self.entries[self.current_index].rois or [])
        self.store_roi(index, rect)
        self.create_roi_item(rect, index)

    def create_roi_item(self, rect: QRectF, roi_index: int):
        item = RoiItem(
            rect,
            self.view.image_bounds,
            self.result_text(roi_index),
            lambda changed, index=roi_index: self.store_roi(index, changed),
            self.view.aspect_ratio,
        )
        self.roi_items.append(item)
        self.scene.addItem(item)

    def set_aspect_ratio(self, value: float):
        self.view.aspect_ratio = value
        for item in self.roi_items:
            item.aspect_ratio = value

    def store_roi(self, roi_index: int, rect: QRectF):
        if self.current_index < 0:
            return
        r = rect.normalized().intersected(self.view.image_bounds)
        x1, y1, x2, y2 = round(r.left()), round(r.top()), round(r.right()), round(r.bottom())
        if x2 <= x1 or y2 <= y1:
            return
        rois = self.entries[self.current_index].rois
        assert rois is not None
        value = (x1, y1, x2 - x1, y2 - y1)
        old = rois[roi_index] if roi_index < len(rois) else None
        if roi_index == len(rois):
            rois.append(value)
        elif roi_index < len(rois):
            rois[roi_index] = value
        if old != value:
            self.results.pop(self.result_key(self.entries[self.current_index], roi_index), None)
            if roi_index < len(self.roi_items):
                self.roi_items[roi_index].label = self.roi_label(roi_index)
                self.roi_items[roi_index].update()
        self.update_status()

    def remove_roi_items(self):
        for item in self.roi_items:
            self.scene.removeItem(item)
        self.roi_items.clear()

    def clear_rois(self):
        if self.current_index < 0:
            return
        entry = self.entries[self.current_index]
        self.clear_results_for(entry)
        entry.rois = []
        self.remove_roi_items()
        self.update_status()

    def clear_all_images(self):
        if not self.entries:
            return
        if QMessageBox.question(self, "Clear all images?", "Remove every image from this batch?") != QMessageBox.StandardButton.Yes:
            return
        self.remove_roi_items()
        self.entries.clear()
        self.results.clear()
        self.image_list.clear()
        self.current_index = -1
        self.pixmap_item.setPixmap(QPixmap())
        self.view.image_bounds = QRectF()
        self.scene.setSceneRect(QRectF())
        self.update_status()

    def apply_roi_to_all(self):
        if self.current_index < 0 or not self.entries[self.current_index].rois:
            QMessageBox.information(self, "No ROI", "Draw at least one ROI first.")
            return
        rois = list(self.entries[self.current_index].rois or [])
        incompatible = [
            entry.path.name
            for entry in self.entries
            if any(x + w > entry.size[0] or y + h > entry.size[1] for x, y, w, h in rois)
        ]
        if incompatible:
            QMessageBox.warning(self, "ROI does not fit all images", "Outside image bounds:\n" + "\n".join(incompatible[:12]))
            return
        for entry in self.entries:
            self.clear_results_for(entry)
            entry.rois = list(rois)
        self.select_image(self.current_index)

    def choose_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Choose trained classification model", "", "PyTorch model (*.pt)")
        if filename:
            self.model_path = Path(filename)
            self.model = None
            self.results.clear()
            self.update_model_label()
            if self.current_index >= 0:
                self.select_image(self.current_index)

    def classify(self, all_images: bool):
        targets = self.entries if all_images else ([self.entries[self.current_index]] if self.current_index >= 0 else [])
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
        """Use the same resize and normalization as the custom training dataset."""
        assert self.model is not None
        core = self.model.model
        imgsz_value = getattr(core, "args", {}).get("imgsz", 224)
        imgsz = int(imgsz_value[0] if isinstance(imgsz_value, (tuple, list)) else imgsz_value)
        preprocessing = transforms.Compose(
            [
                transforms.Resize((imgsz, imgsz), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        batch = torch.stack([preprocessing(image) for image in images])
        device = next(core.parameters()).device
        batch = batch.to(device)
        core.eval()
        with torch.inference_mode():
            output = core(batch)
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output.cpu()

    def result_text(self, roi_index: int) -> str:
        if self.current_index < 0:
            return self.roi_label(roi_index)
        result = self.results.get(self.result_key(self.entries[self.current_index], roi_index))
        return self.roi_label(roi_index) if result is None else f"{result[0]}  {result[1]:.1%}"

    def result_key(self, entry: ImageEntry, roi_index: int) -> tuple[Path, int]:
        return entry.path.resolve(), roi_index

    def clear_results_for(self, entry: ImageEntry):
        path = entry.path.resolve()
        self.results = {key: value for key, value in self.results.items() if key[0] != path}

    def update_model_label(self):
        self.model_label.setText(f"Model: {self.model_path}" if self.model_path else "Model: not selected")

    def update_status(self, message: str | None = None):
        if message:
            self.status.setText(message)
        elif self.current_index < 0:
            self.status.setText("Open images, draw one or more ROIs, then choose a trained model.")
        else:
            entry = self.entries[self.current_index]
            count = len(entry.rois or [])
            classified = sum(self.result_key(entry, i) in self.results for i in range(count))
            self.status.setText(
                f"{self.current_index + 1} / {len(self.entries)} — {entry.path.name} — "
                f"{count} ROI(s), {classified} classified"
            )

    @staticmethod
    def roi_label(index: int) -> str:
        label = ""
        value = index + 1
        while value:
            value, remainder = divmod(value - 1, 26)
            label = chr(ord("A") + remainder) + label
        return label


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Trained Ultralytics classification checkpoint (best.pt)")
    return parser.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    window = ClassificationWindow(args.model.resolve() if args.model else None)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
