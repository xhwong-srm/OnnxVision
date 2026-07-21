"""Keyboard-friendly image picker and mover.

Install:  pip install PySide6
Run:      python image_mover.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


SUPPORTED_SUFFIXES = {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Picker and Mover")
        self.resize(1200, 780)

        self.paths: list[Path] = []
        self.selected: set[Path] = set()
        self.current_pixmap = QPixmap()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = QVBoxLayout()
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.show_image)
        self.image_list.itemDoubleClicked.connect(lambda _item: self.toggle_current())
        left.addWidget(self.image_list, 1)

        add_folder = QPushButton("Add folder...")
        add_folder.clicked.connect(self.add_folder)
        left.addWidget(add_folder)

        clear = QPushButton("Clear list")
        clear.clicked.connect(self.clear_list)
        left.addWidget(clear)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        self.image_label = QLabel("Add a folder to begin")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("QLabel { background: #202020; color: #dddddd; }")
        right.addWidget(self.image_label, 1)

        self.status = QLabel()
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.status)

        controls = QHBoxLayout()
        previous = QPushButton("Previous  (Left)")
        previous.clicked.connect(lambda: self.navigate(-1))
        controls.addWidget(previous)

        select = QPushButton("Select / unselect  (Enter)")
        select.clicked.connect(self.toggle_current)
        controls.addWidget(select)

        following = QPushButton("Next  (Right)")
        following.clicked.connect(lambda: self.navigate(1))
        controls.addWidget(following)
        right.addLayout(controls)

        move = QPushButton("Move selected images...")
        move.clicked.connect(self.move_selected)
        right.addWidget(move)
        layout.addLayout(right, 4)

        self.add_shortcut(Qt.Key.Key_Left, lambda: self.navigate(-1))
        self.add_shortcut(Qt.Key.Key_Right, lambda: self.navigate(1))
        self.add_shortcut(Qt.Key.Key_Up, lambda: self.navigate(-1))
        self.add_shortcut(Qt.Key.Key_Down, lambda: self.navigate(1))
        self.add_shortcut(Qt.Key.Key_PageUp, lambda: self.navigate(-10))
        self.add_shortcut(Qt.Key.Key_PageDown, lambda: self.navigate(10))
        self.add_shortcut(Qt.Key.Key_Return, self.toggle_current)
        self.add_shortcut(Qt.Key.Key_Enter, self.toggle_current)
        self.update_status()

    def add_shortcut(self, key: Qt.Key, action) -> None:
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut.activated.connect(action)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select an image folder")
        if not folder:
            return

        known = {path.resolve() for path in self.paths}
        new_paths = sorted(
            path
            for path in Path(folder).iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and path.resolve() not in known
        )
        for path in new_paths:
            self.paths.append(path)
            self.image_list.addItem(QListWidgetItem(path.name))

        if self.image_list.currentRow() < 0 and self.paths:
            self.image_list.setCurrentRow(0)
        self.update_status()

    def show_image(self, index: int) -> None:
        if not 0 <= index < len(self.paths):
            self.current_pixmap = QPixmap()
            self.image_label.clear()
            self.update_status()
            return

        self.current_pixmap = QPixmap(str(self.paths[index]))
        if self.current_pixmap.isNull():
            self.image_label.setText(f"Could not display\n{self.paths[index].name}")
        else:
            self.scale_current_image()
        self.update_status()

    def scale_current_image(self) -> None:
        if self.current_pixmap.isNull():
            return
        size = self.image_label.contentsRect().size()
        self.image_label.setPixmap(
            self.current_pixmap.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.scale_current_image()

    def navigate(self, amount: int) -> None:
        if not self.paths:
            return
        current = max(self.image_list.currentRow(), 0)
        self.image_list.setCurrentRow(min(max(current + amount, 0), len(self.paths) - 1))

    def toggle_current(self) -> None:
        index = self.image_list.currentRow()
        if not 0 <= index < len(self.paths):
            return

        path = self.paths[index]
        if path in self.selected:
            self.selected.remove(path)
        else:
            self.selected.add(path)
        self.refresh_item(index)
        self.update_status()

        if index < len(self.paths) - 1:
            self.image_list.setCurrentRow(index + 1)

    def refresh_item(self, index: int) -> None:
        path = self.paths[index]
        item = self.image_list.item(index)
        item.setText(("✓  " if path in self.selected else "   ") + path.name)
        item.setBackground(Qt.GlobalColor.darkGreen if path in self.selected else Qt.GlobalColor.transparent)

    def clear_list(self) -> None:
        if self.paths and QMessageBox.question(self, "Clear list?", "Remove all images from this list?") != QMessageBox.StandardButton.Yes:
            return
        self.paths.clear()
        self.selected.clear()
        self.image_list.clear()
        self.current_pixmap = QPixmap()
        self.image_label.setText("Add a folder to begin")
        self.update_status()

    def move_selected(self) -> None:
        targets = [path for path in self.paths if path in self.selected]
        if not targets:
            QMessageBox.information(self, "Nothing selected", "Select one or more images first.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Move selected images to...")
        if not folder:
            return
        destination = Path(folder).resolve()
        if QMessageBox.question(
            self,
            "Move images?",
            f"Move {len(targets)} selected image(s) to:\n{destination}?",
        ) != QMessageBox.StandardButton.Yes:
            return

        moved: set[Path] = set()
        failures: list[str] = []
        for source in targets:
            target = destination / source.name
            try:
                if source.parent.resolve() == destination:
                    raise FileExistsError("source is already in the destination folder")
                if target.exists():
                    raise FileExistsError("a file with this name already exists at the destination")
                shutil.move(str(source), str(target))
                moved.add(source)
            except Exception as error:
                failures.append(f"{source.name}: {error}")

        if moved:
            self.paths = [path for path in self.paths if path not in moved]
            self.selected.difference_update(moved)
            self.rebuild_list()

        message = f"Moved {len(moved)} of {len(targets)} image(s)."
        if failures:
            QMessageBox.warning(self, "Move completed with errors", message + "\n\n" + "\n".join(failures[:20]))
        else:
            QMessageBox.information(self, "Move complete", message)

    def rebuild_list(self) -> None:
        old_row = self.image_list.currentRow()
        self.image_list.clear()
        for index, path in enumerate(self.paths):
            self.image_list.addItem(QListWidgetItem(path.name))
            self.refresh_item(index)
        if self.paths:
            self.image_list.setCurrentRow(min(max(old_row, 0), len(self.paths) - 1))
        else:
            self.current_pixmap = QPixmap()
            self.image_label.setText("Add a folder to begin")
            self.update_status()

    def update_status(self) -> None:
        index = self.image_list.currentRow()
        position = f"{index + 1} / {len(self.paths)}" if index >= 0 else f"0 / {len(self.paths)}"
        self.status.setText(f"{position}    •    {len(self.selected)} selected    •    Enter selects and advances")


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
