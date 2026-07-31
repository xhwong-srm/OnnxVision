"""Compare native RF-DETR Nano inference with the exported OnnxVision model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from rfdetr import RFDETRNano


DEFAULT_IMAGE_URL = "https://media.roboflow.com/dog.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/rfdetr-nano-detection-c24.onnx"))
    parser.add_argument("--image", help=f"Local image or URL (default: {DEFAULT_IMAGE_URL})")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rfdetr-nano-experiment"))
    return parser.parse_args()


def load_image(source: str | None, output_dir: Path) -> tuple[Image.Image, Path]:
    source = source or DEFAULT_IMAGE_URL
    if source.startswith(("http://", "https://")):
        image_path = output_dir / "source.jpg"
        with urlopen(source) as response:
            image_path.write_bytes(response.read())
    else:
        image_path = Path(source).expanduser().resolve()
    return Image.open(image_path).convert("RGB"), image_path


def read_names(session: ort.InferenceSession) -> list[str]:
    raw = session.get_modelmeta().custom_metadata_map["names"]
    mapping = json.loads(raw)
    return [mapping[str(index)] for index in range(len(mapping))]


def onnx_predict(
    session: ort.InferenceSession, image: Image.Image, names: list[str], threshold: float
) -> list[dict[str, object]]:
    input_metadata = session.get_inputs()[0]
    if input_metadata.shape[1] == 1:
        value = np.asarray(image.convert("L"), dtype=np.uint8)[None, None]
    else:
        value = np.asarray(image, dtype=np.uint8)[..., ::-1].copy()[None]
    output_values = session.run(None, {input_metadata.name: value})
    outputs = {item.name: value for item, value in zip(session.get_outputs(), output_values)}
    height, width = image.height, image.width
    predictions = []
    for box, score, class_id in zip(outputs["boxes"][0], outputs["scores"][0], outputs["class_ids"][0]):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box
        predictions.append({
            "class_id": int(class_id),
            "class_name": names[int(class_id)],
            "confidence": float(score),
            "xyxy": [float(x1 * width), float(y1 * height), float(x2 * width), float(y2 * height)],
        })
    return predictions


def native_predict(model: RFDETRNano, image: Image.Image, threshold: float) -> list[dict[str, object]]:
    detections = model.predict(image, threshold=threshold)
    return [{
        "class_id": int(class_id),
        "class_name": str(class_name),
        "confidence": float(confidence),
        "xyxy": [float(value) for value in box],
    } for box, confidence, class_id, class_name in zip(
        detections.xyxy,
        detections.confidence,
        detections.class_id,
        detections.data["class_name"],
    )]


def annotate(image: Image.Image, predictions: list[dict[str, object]], output: Path) -> None:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for prediction in predictions:
        box = prediction["xyxy"]
        label = f"{prediction['class_name']} {prediction['confidence']:.3f}"
        draw.rectangle(box, outline=(255, 40, 40), width=3)
        draw.text((box[0] + 3, box[1] + 3), label, fill=(255, 40, 40))
    result.save(output)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image, image_path = load_image(args.image, args.output_dir)

    native_model = RFDETRNano()
    native = native_predict(native_model, image, args.threshold)
    session = ort.InferenceSession(str(args.model.resolve()), providers=["CPUExecutionProvider"])
    names = read_names(session)
    onnx = onnx_predict(session, image, names, args.threshold)

    annotate(image, native, args.output_dir / "native.jpg")
    annotate(image, onnx, args.output_dir / "onnx.jpg")
    report = {
        "image": str(image_path),
        "model": str(args.model.resolve()),
        "threshold": args.threshold,
        "native": native,
        "onnx": onnx,
        "same_class_sequence": [item["class_name"] for item in native]
        == [item["class_name"] for item in onnx],
        "maximum_box_error_pixels": max(
            (
                abs(a - b)
                for native_item, onnx_item in zip(native, onnx)
                for a, b in zip(native_item["xyxy"], onnx_item["xyxy"])
            ),
            default=0.0,
        ),
        "maximum_score_error": max(
            (abs(a["confidence"] - b["confidence"]) for a, b in zip(native, onnx)),
            default=0.0,
        ),
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
