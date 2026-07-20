"""Keyboard-first builder for Ultralytics YOLO classification datasets.

Run:
    uv run python classification_dataset_builder.py
    uv run python classification_dataset_builder.py --model runs/classify/.../best.pt

Workflow:
    1. Add images or a folder.
    2. Add/select classes. Tab cycles classes; Enter assigns and advances.
    3. Optionally load a trained classification model and auto-label images.
    4. Export copies to ``train/class``, ``val/class`` and ``test/class``.
"""

from __future__ import annotations

import hashlib
import random
import shutil
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO


SUPPORTED_SUFFIXES = {".bmp", ".dng", ".gif", ".heic", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
INVALID_CLASS_CHARACTERS = set('<>:"/\\|?*')
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def validate_class_name(value: str) -> str:
    name = value.strip()
    if not name or name in {".", ".."}:
        raise ValueError("Class name cannot be empty.")
    if any(character in INVALID_CLASS_CHARACTERS for character in name) or name.endswith((".", " ")):
        raise ValueError('Class names cannot contain <>:"/\\|?* or end with a dot or space.')
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


def balance_training_groups(
    groups: dict[str, list[Path]], strategy: str, rng: random.Random
) -> dict[str, list[Path]]:
    """Return equal-sized training classes without changing the input lists."""
    if not groups or any(not images for images in groups.values()):
        return {name: list(images) for name, images in groups.items()}
    target = max(map(len, groups.values())) if strategy == "Oversample" else min(map(len, groups.values()))
    balanced: dict[str, list[Path]] = {}
    for name, images in groups.items():
        selected = list(images)
        if strategy == "Oversample":
            selected.extend(rng.choice(images) for _ in range(target - len(images)))
        else:
            rng.shuffle(selected)
            selected = selected[:target]
        balanced[name] = selected
    return balanced


class MainWindow(QMainWindow):
    def __init__(self, initial_model: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Classification Dataset Builder")
        self.resize(1300, 820)
        self.paths: list[Path] = []
        self.assignments: dict[Path, str] = {}
        self.confidences: dict[Path, float] = {}
        self.classes: list[str] = []
        self.current_pixmap = QPixmap()
        self.model_path = initial_model
        self.model: YOLO | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        images_panel = QVBoxLayout()
        images_panel.addWidget(QLabel("Images"))
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.show_image)
        images_panel.addWidget(self.image_list, 1)
        add_images = QPushButton("Add images...")
        add_images.clicked.connect(self.add_images)
        images_panel.addWidget(add_images)
        add_folder = QPushButton("Add folder...")
        add_folder.clicked.connect(self.add_folder)
        images_panel.addWidget(add_folder)
        load_dataset = QPushButton("Load YOLO dataset...")
        load_dataset.clicked.connect(self.load_dataset)
        images_panel.addWidget(load_dataset)
        clear = QPushButton("Clear images")
        clear.clicked.connect(self.clear_images)
        images_panel.addWidget(clear)
        layout.addLayout(images_panel, 1)

        center = QVBoxLayout()
        self.image_label = QLabel("Add images to begin")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(480, 320)
        self.image_label.setStyleSheet("QLabel { background: #202020; color: #dddddd; }")
        center.addWidget(self.image_label, 1)
        self.assignment_label = QLabel()
        self.assignment_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(self.assignment_label)

        navigation = QHBoxLayout()
        previous = QPushButton("Previous  (Left)")
        previous.clicked.connect(lambda: self.navigate(-1))
        navigation.addWidget(previous)
        assign = QPushButton("Assign selected class + next  (Enter)")
        assign.clicked.connect(self.assign_and_advance)
        navigation.addWidget(assign)
        following = QPushButton("Next  (Right)")
        following.clicked.connect(lambda: self.navigate(1))
        navigation.addWidget(following)
        center.addLayout(navigation)

        model_row = QHBoxLayout()
        choose_model = QPushButton("Choose model...")
        choose_model.clicked.connect(self.choose_model)
        model_row.addWidget(choose_model)
        self.model_label = QLabel()
        self.model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        model_row.addWidget(self.model_label, 1)
        center.addLayout(model_row)
        auto_row = QHBoxLayout()
        auto_current = QPushButton("Auto-label current")
        auto_current.clicked.connect(lambda: self.auto_label(False))
        auto_row.addWidget(auto_current)
        auto_all = QPushButton("Auto-label all")
        auto_all.clicked.connect(lambda: self.auto_label(True))
        auto_row.addWidget(auto_all)
        center.addLayout(auto_row)
        self.status = QLabel()
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(self.status)
        layout.addLayout(center, 4)

        classes_panel = QVBoxLayout()
        classes_panel.addWidget(QLabel("Classes (Tab cycles)"))
        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(lambda _row: self.update_status())
        self.class_list.itemDoubleClicked.connect(lambda _item: self.assign_and_advance())
        classes_panel.addWidget(self.class_list, 1)
        add_class = QPushButton("Add class...")
        add_class.clicked.connect(lambda _checked=False: self.add_class())
        classes_panel.addWidget(add_class)
        remove_class = QPushButton("Remove class")
        remove_class.clicked.connect(self.remove_class)
        classes_panel.addWidget(remove_class)

        classes_panel.addWidget(QLabel("Export split"))
        split_row = QHBoxLayout()
        self.train_ratio = self.create_ratio_spinbox(70.0)
        self.val_ratio = self.create_ratio_spinbox(20.0)
        self.test_ratio = self.create_ratio_spinbox(10.0)
        for label, widget in (("Train", self.train_ratio), ("Val", self.val_ratio), ("Test", self.test_ratio)):
            column = QVBoxLayout()
            column.addWidget(QLabel(label))
            column.addWidget(widget)
            split_row.addLayout(column)
        classes_panel.addLayout(split_row)
        self.balance_training = QCheckBox("Balance training split")
        self.balance_training.setToolTip(
            "Validation and test are not balanced, keeping evaluation data free from duplicated samples."
        )
        classes_panel.addWidget(self.balance_training)
        self.balance_strategy = QComboBox()
        self.balance_strategy.addItems(["Oversample", "Undersample"])
        self.balance_strategy.setToolTip(
            "Oversample duplicates minority training images; undersample discards majority training images."
        )
        classes_panel.addWidget(self.balance_strategy)
        export = QPushButton("Generate dataset...")
        export.clicked.connect(self.export_dataset)
        classes_panel.addWidget(export)
        layout.addLayout(classes_panel, 1)

        self.add_shortcut(Qt.Key.Key_Left, lambda: self.navigate(-1))
        self.add_shortcut(Qt.Key.Key_Right, lambda: self.navigate(1))
        self.add_shortcut(Qt.Key.Key_Tab, lambda: self.cycle_class(1))
        self.add_shortcut(Qt.Key.Key_Backtab, lambda: self.cycle_class(-1))
        self.add_shortcut(Qt.Key.Key_Return, self.assign_and_advance)
        self.add_shortcut(Qt.Key.Key_Enter, self.assign_and_advance)
        self.update_model_label()
        self.update_status()

    @staticmethod
    def create_ratio_spinbox(value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0, 100.0)
        box.setDecimals(0)
        box.setSuffix("%")
        box.setValue(value)
        return box

    def add_shortcut(self, key: Qt.Key, action) -> None:
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut.activated.connect(action)

    def add_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add images", "", "Images (*.bmp *.dng *.gif *.heic *.jpeg *.jpg *.mpo *.png *.tif *.tiff *.webp)")
        self.add_paths([Path(filename) for filename in files])

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add image folder")
        if folder:
            self.add_paths(sorted(path for path in Path(folder).iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES))

    def load_dataset(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select YOLO classification dataset")
        if not folder:
            return
        root = Path(folder)
        split_dirs = [root / name for name in ("train", "val", "test") if (root / name).is_dir()]
        if not split_dirs:
            QMessageBox.warning(
                self,
                "Not a YOLO classification dataset",
                "The selected folder must contain train, val, or test directories.",
            )
            return

        labeled_paths: list[tuple[Path, str]] = []
        invalid_classes: list[str] = []
        for split_dir in split_dirs:
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                try:
                    class_name = validate_class_name(class_dir.name)
                except ValueError:
                    invalid_classes.append(str(class_dir))
                    continue
                for path in sorted(class_dir.rglob("*")):
                    if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                        labeled_paths.append((path, class_name))
        if not labeled_paths:
            QMessageBox.warning(self, "Dataset is empty", "No supported images were found in class folders.")
            return

        canonical = {name.casefold(): name for name in self.classes}
        for _path, class_name in labeled_paths:
            if class_name.casefold() not in canonical:
                canonical[class_name.casefold()] = class_name
                self.classes.append(class_name)
                self.class_list.addItem(class_name)
        self.add_paths([path for path, _class_name in labeled_paths])
        stored_paths = {path.resolve(): path for path in self.paths}
        for path, class_name in labeled_paths:
            stored = stored_paths.get(path.resolve())
            if stored is not None:
                self.assignments[stored] = canonical[class_name.casefold()]
                self.confidences.pop(stored, None)
        self.refresh_all_items()
        message = f"Loaded {len(labeled_paths)} labeled image(s) from {len(split_dirs)} split folder(s)."
        if invalid_classes:
            message += f" Skipped {len(invalid_classes)} invalid class folder(s)."
        self.update_status(message)

    def add_paths(self, paths: list[Path]) -> None:
        known = {path.resolve() for path in self.paths}
        for path in paths:
            resolved = path.resolve()
            if resolved not in known and path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                self.paths.append(path)
                self.image_list.addItem(QListWidgetItem(path.name))
                known.add(resolved)
        if self.image_list.currentRow() < 0 and self.paths:
            self.image_list.setCurrentRow(0)
        self.refresh_all_items()

    def add_class(self, suggested: str = "") -> str | None:
        value, accepted = QInputDialog.getText(
            self,
            "Add class",
            "Class name:",
            QLineEdit.EchoMode.Normal,
            suggested,
        )
        if not accepted:
            return None
        try:
            name = validate_class_name(value)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid class name", str(error))
            return None
        existing = next((item for item in self.classes if item.casefold() == name.casefold()), None)
        if existing is not None:
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
        name = self.classes[row]
        count = sum(label == name for label in self.assignments.values())
        prompt = f"Remove class '{name}'?"
        if count:
            prompt += f"\n\n{count} assigned image(s) will become unassigned."
        if QMessageBox.question(self, "Remove class?", prompt) != QMessageBox.StandardButton.Yes:
            return
        self.classes.pop(row)
        self.class_list.takeItem(row)
        for path in [path for path, label in self.assignments.items() if label == name]:
            self.assignments.pop(path, None)
            self.confidences.pop(path, None)
        if self.classes:
            self.class_list.setCurrentRow(min(row, len(self.classes) - 1))
        self.refresh_all_items()

    def cycle_class(self, amount: int) -> None:
        if not self.classes:
            return
        current = self.class_list.currentRow()
        self.class_list.setCurrentRow((current + amount) % len(self.classes))

    def assign_and_advance(self) -> None:
        image_row = self.image_list.currentRow()
        class_row = self.class_list.currentRow()
        if not 0 <= image_row < len(self.paths):
            return
        if not 0 <= class_row < len(self.classes):
            QMessageBox.information(self, "Class required", "Add and select a class first.")
            return
        path = self.paths[image_row]
        self.assignments[path] = self.classes[class_row]
        self.confidences.pop(path, None)
        self.refresh_item(image_row)
        if image_row < len(self.paths) - 1:
            self.image_list.setCurrentRow(image_row + 1)
        else:
            self.update_status("All images reviewed. You can export the dataset.")

    def navigate(self, amount: int) -> None:
        if self.paths:
            current = max(self.image_list.currentRow(), 0)
            self.image_list.setCurrentRow(min(max(current + amount, 0), len(self.paths) - 1))

    def show_image(self, index: int) -> None:
        if not 0 <= index < len(self.paths):
            self.current_pixmap = QPixmap()
            self.image_label.setText("Add images to begin")
            self.update_status()
            return
        self.current_pixmap = QPixmap(str(self.paths[index]))
        if self.current_pixmap.isNull():
            self.image_label.setText(f"Could not display\n{self.paths[index].name}")
        else:
            self.scale_current_image()
        assigned = self.assignments.get(self.paths[index])
        if assigned in self.classes:
            self.class_list.setCurrentRow(self.classes.index(assigned))
        self.update_status()

    def scale_current_image(self) -> None:
        if not self.current_pixmap.isNull():
            self.image_label.setPixmap(self.current_pixmap.scaled(self.image_label.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.scale_current_image()

    def refresh_item(self, index: int) -> None:
        path = self.paths[index]
        label = self.assignments.get(path)
        confidence = self.confidences.get(path)
        suffix = f" ({confidence:.1%})" if confidence is not None else ""
        self.image_list.item(index).setText(f"{label or '—'}{suffix}  |  {path.name}")
        self.image_list.item(index).setBackground(Qt.GlobalColor.darkGreen if label else Qt.GlobalColor.transparent)

    def refresh_all_items(self) -> None:
        for index in range(len(self.paths)):
            self.refresh_item(index)
        self.update_status()

    def clear_images(self) -> None:
        if self.paths and QMessageBox.question(self, "Clear images?", "Remove all images and their assignments from this session?") != QMessageBox.StandardButton.Yes:
            return
        self.paths.clear()
        self.assignments.clear()
        self.confidences.clear()
        self.image_list.clear()
        self.current_pixmap = QPixmap()
        self.image_label.setText("Add images to begin")
        self.update_status()

    def choose_model(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Choose trained classification model", "", "PyTorch model (*.pt)")
        if filename:
            self.model_path = Path(filename)
            self.model = None
            self.update_model_label()

    def auto_label(self, all_images: bool) -> None:
        row = self.image_list.currentRow()
        targets = self.paths if all_images else ([self.paths[row]] if 0 <= row < len(self.paths) else [])
        if not targets:
            QMessageBox.information(self, "No images", "Add at least one image first.")
            return
        if not self.model_path or not self.model_path.is_file():
            self.choose_model()
            if not self.model_path or not self.model_path.is_file():
                return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.status.setText(f"Auto-labeling {len(targets)} full image(s)...")
        QApplication.processEvents()
        try:
            if self.model is None:
                self.model = YOLO(str(self.model_path), task="classify")
            results = self.model([str(path) for path in targets], verbose=False)
            predicted = []
            for path, result in zip(targets, results):
                if result.probs is None:
                    raise ValueError("The selected checkpoint did not return classification probabilities.")
                class_id = int(result.probs.top1)
                name = validate_class_name(str(result.names[class_id]))
                predicted.append((path, name, float(result.probs.top1conf)))
            for _path, name, _confidence in predicted:
                if not any(existing.casefold() == name.casefold() for existing in self.classes):
                    self.classes.append(name)
                    self.class_list.addItem(name)
            canonical = {name.casefold(): name for name in self.classes}
            for path, name, confidence in predicted:
                self.assignments[path] = canonical[name.casefold()]
                self.confidences[path] = confidence
            self.refresh_all_items()
            if row >= 0:
                self.show_image(row)
            self.update_status(f"Auto-labeled {len(predicted)} image(s). Review predictions before export.")
        except Exception as error:
            QMessageBox.critical(self, "Auto-labeling failed", str(error))
            self.update_status()
        finally:
            QApplication.restoreOverrideCursor()

    def export_dataset(self) -> None:
        if not self.paths:
            QMessageBox.information(self, "No images", "Add and label images first.")
            return
        missing = [path for path in self.paths if path not in self.assignments]
        if missing:
            QMessageBox.warning(self, "Unassigned images", f"Assign every image before export.\n\nUnassigned: {len(missing)}")
            return
        values = (self.train_ratio.value(), self.val_ratio.value(), self.test_ratio.value())
        if abs(sum(values) - 100.0) > 0.01:
            QMessageBox.warning(self, "Invalid split", "Train, val, and test percentages must add up to 100%.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose the parent folder for the dataset")
        if not folder:
            return
        name, accepted = QInputDialog.getText(
            self,
            "Dataset folder",
            "New dataset folder name:",
            QLineEdit.EchoMode.Normal,
            "classification_dataset",
        )
        if not accepted:
            return
        try:
            dataset_name = validate_class_name(name)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid folder name", str(error))
            return
        output = Path(folder) / dataset_name
        if output.exists():
            QMessageBox.warning(self, "Destination exists", f"Choose a new folder name. Nothing was changed:\n{output}")
            return
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in self.paths:
            grouped[self.assignments[path]].append(path)
        rng = random.Random(42)
        ratios = tuple(value / 100.0 for value in values)
        copied = 0
        try:
            split_groups: dict[str, dict[str, list[Path]]] = {
                split: {class_name: [] for class_name in grouped}
                for split in ("train", "val", "test")
            }
            for class_name, images in grouped.items():
                rng.shuffle(images)
                counts = split_counts(len(images), ratios)  # type: ignore[arg-type]
                offset = 0
                for split, count in zip(("train", "val", "test"), counts):
                    split_groups[split][class_name] = images[offset : offset + count]
                    offset += count
            if self.balance_training.isChecked():
                split_groups["train"] = balance_training_groups(
                    split_groups["train"], self.balance_strategy.currentText(), rng
                )

            for split, classes in split_groups.items():
                for class_name, images in classes.items():
                    destination_dir = output / split / class_name
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    for copy_index, source in enumerate(images):
                        destination = destination_dir / source.name
                        if destination.exists():
                            digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
                            destination = destination_dir / f"{source.stem}_{digest}_{copy_index:04d}{source.suffix.lower()}"
                            collision = 1
                            while destination.exists():
                                destination = destination_dir / f"{source.stem}_{digest}_{copy_index:04d}_{collision}{source.suffix.lower()}"
                                collision += 1
                        shutil.copy2(source, destination)
                        copied += 1
        except Exception as error:
            QMessageBox.critical(self, "Export failed", f"Copied {copied} image(s) before the error.\n\n{error}\n\nPartial output: {output}")
            return
        QMessageBox.information(self, "Dataset created", f"Created a YOLO classification dataset with {copied} image(s):\n{output}")
        self.update_status(f"Dataset created: {output}")

    def update_model_label(self) -> None:
        self.model_label.setText(f"Model: {self.model_path}" if self.model_path else "Model: not selected")

    def update_status(self, message: str | None = None) -> None:
        if message:
            self.status.setText(message)
            return
        row = self.image_list.currentRow()
        assigned_count = len(self.assignments)
        position = f"{row + 1} / {len(self.paths)}" if row >= 0 else f"0 / {len(self.paths)}"
        selected = self.classes[self.class_list.currentRow()] if 0 <= self.class_list.currentRow() < len(self.classes) else "none"
        current = self.assignments.get(self.paths[row], "unassigned") if 0 <= row < len(self.paths) else "unassigned"
        confidence = self.confidences.get(self.paths[row]) if 0 <= row < len(self.paths) else None
        confidence_text = f" ({confidence:.1%})" if confidence is not None else ""
        self.assignment_label.setText(f"Current: {current}{confidence_text}    •    Selected class: {selected}")
        self.status.setText(f"{position}    •    {assigned_count} / {len(self.paths)} assigned    •    Tab class, Enter assign + next")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Trained Ultralytics classification checkpoint (best.pt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args.model.resolve() if args.model else None)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
