using System;
using System.Collections.Generic;
using System.Drawing;
using Euresys.Open_eVision_22_12;
using OnnxVision.Classification;
using OnnxVision.Detection;
using OnnxVision.Imaging;

namespace OnnxVision.Euresys
{
    public static class EuresysOnnxExtensions
    {
        public static OnnxClassification Classify(this OnnxClassificationModel model, EROIBW8 roi)
        {
            if (model == null)
                throw new ArgumentNullException("model");
            if (roi == null)
                throw new ArgumentNullException("roi");
            return model.Classify(CreateImageBuffer(roi));
        }

        public static OnnxClassification Classify(this OnnxClassificationModel model, EROIC24 roi)
        {
            if (model == null)
                throw new ArgumentNullException("model");
            if (roi == null)
                throw new ArgumentNullException("roi");
            return model.Classify(CreateImageBuffer(roi));
        }

        public static OnnxClassification Classify(this OnnxClassificationModel model, EImageBW8 image)
        {
            if (image == null)
                throw new ArgumentNullException("image");
            return model.Classify(image, new Rectangle(0, 0, image.Width, image.Height));
        }

        public static OnnxClassification Classify(this OnnxClassificationModel model,
            EImageBW8 image, Rectangle region)
        {
            if (model == null)
                throw new ArgumentNullException("model");
            return model.Classify(CreateImageBuffer(image, region));
        }

        public static OnnxClassification Classify(this OnnxClassificationModel model, EImageC24 image)
        {
            if (image == null)
                throw new ArgumentNullException("image");
            return model.Classify(image, new Rectangle(0, 0, image.Width, image.Height));
        }

        public static OnnxClassification Classify(this OnnxClassificationModel model,
            EImageC24 image, Rectangle region)
        {
            if (model == null)
                throw new ArgumentNullException("model");
            return model.Classify(CreateImageBuffer(image, region));
        }

        public static IReadOnlyList<OnnxDetection> Detect(this OnnxObjectDetectionModel model,
            string imagePath, float confidenceThreshold = 0.5f, float nmsIouThreshold = 1.0f)
        {
            if (model == null)
                throw new ArgumentNullException("model");
            if (string.IsNullOrWhiteSpace(imagePath))
                throw new ArgumentException("An image path is required.", "imagePath");

            if (model.RequiresColorInput)
            {
                using (var image = new EImageC24())
                {
                    image.Load(imagePath);
                    return model.Detect(image, confidenceThreshold, nmsIouThreshold);
                }
            }

            using (var bw8Image = new EImageBW8())
            {
                bw8Image.Load(imagePath);
                return model.Detect(bw8Image, confidenceThreshold, nmsIouThreshold);
            }
        }

        public static IReadOnlyList<OnnxDetection> Detect(this OnnxObjectDetectionModel model,
            EImageBW8 image, float confidenceThreshold = 0.5f, float nmsIouThreshold = 1.0f)
        {
            if (image == null)
                throw new ArgumentNullException("image");
            return model.Detect(image, new Rectangle(0, 0, image.Width, image.Height),
                confidenceThreshold, nmsIouThreshold);
        }

        public static IReadOnlyList<OnnxDetection> Detect(this OnnxObjectDetectionModel model,
            EImageBW8 image, Rectangle region, float confidenceThreshold = 0.5f,
            float nmsIouThreshold = 1.0f)
        {
            if (model == null)
                throw new ArgumentNullException("model");
            return model.Detect(CreateImageBuffer(image, region),
                confidenceThreshold, nmsIouThreshold);
        }

        public static IReadOnlyList<OnnxDetection> Detect(this OnnxObjectDetectionModel model,
            EImageC24 image, float confidenceThreshold = 0.5f, float nmsIouThreshold = 1.0f)
        {
            if (image == null)
                throw new ArgumentNullException("image");
            return model.Detect(image, new Rectangle(0, 0, image.Width, image.Height),
                confidenceThreshold, nmsIouThreshold);
        }

        public static IReadOnlyList<OnnxDetection> Detect(this OnnxObjectDetectionModel model,
            EImageC24 image, Rectangle region, float confidenceThreshold = 0.5f,
            float nmsIouThreshold = 1.0f)
        {
            if (model == null)
                throw new ArgumentNullException("model");
            return model.Detect(CreateImageBuffer(image, region),
                confidenceThreshold, nmsIouThreshold);
        }

        public static IReadOnlyList<OnnxDetection> Detect(this OnnxObjectDetectionModel model,
            EROIBW8 roi, float confidenceThreshold = 0.5f, float nmsIouThreshold = 1.0f)
        {
            if (model == null)
                throw new ArgumentNullException("model");
            if (roi == null)
                throw new ArgumentNullException("roi");
            return model.Detect(CreateImageBuffer(roi), confidenceThreshold, nmsIouThreshold);
        }

        public static IReadOnlyList<OnnxDetection> Detect(this OnnxObjectDetectionModel model,
            EROIC24 roi, float confidenceThreshold = 0.5f, float nmsIouThreshold = 1.0f)
        {
            if (model == null)
                throw new ArgumentNullException("model");
            if (roi == null)
                throw new ArgumentNullException("roi");
            return model.Detect(CreateImageBuffer(roi), confidenceThreshold, nmsIouThreshold);
        }

        public static OnnxImageBuffer CreateImageBuffer(EImageBW8 image, Rectangle region)
        {
            if (image == null)
                throw new ArgumentNullException("image");
            ValidateRegion(image.Width, image.Height, region);
            if (image.ColPitch != 1)
            {
                throw new NotSupportedException(string.Format(
                    "ONNX BW8 input requires packed 1-byte pixels, but the source column pitch is {0}.",
                    image.ColPitch));
            }
            return new OnnxImageBuffer(image.GetImagePtr(region.X, region.Y),
                region.Width, region.Height, image.RowPitch, OnnxPixelFormat.Bw8);
        }

        public static OnnxImageBuffer CreateImageBuffer(EImageC24 image, Rectangle region)
        {
            if (image == null)
                throw new ArgumentNullException("image");
            ValidateRegion(image.Width, image.Height, region);
            if (image.ColPitch != 3)
            {
                throw new NotSupportedException(string.Format(
                    "ONNX C24 input requires packed 3-byte pixels, but the source column pitch is {0}.",
                    image.ColPitch));
            }
            return new OnnxImageBuffer(image.GetImagePtr(region.X, region.Y),
                region.Width, region.Height, image.RowPitch, OnnxPixelFormat.Bgr24);
        }

        public static OnnxImageBuffer CreateImageBuffer(EROIBW8 roi)
        {
            if (roi == null)
                throw new ArgumentNullException("roi");
            ValidatePackedPixels(roi.ColPitch, 1, "BW8");
            return new OnnxImageBuffer(roi.GetImagePtr(0, 0), roi.Width, roi.Height,
                roi.RowPitch, OnnxPixelFormat.Bw8);
        }

        public static OnnxImageBuffer CreateImageBuffer(EROIC24 roi)
        {
            if (roi == null)
                throw new ArgumentNullException("roi");
            ValidatePackedPixels(roi.ColPitch, 3, "C24");
            return new OnnxImageBuffer(roi.GetImagePtr(0, 0), roi.Width, roi.Height,
                roi.RowPitch, OnnxPixelFormat.Bgr24);
        }

        private static void ValidateRegion(int imageWidth, int imageHeight, Rectangle region)
        {
            if (region.Width <= 0 || region.Height <= 0)
                throw new ArgumentException("The ONNX image region must have positive dimensions.", "region");
            if (region.X < 0 || region.Y < 0 ||
                region.X > imageWidth - region.Width || region.Y > imageHeight - region.Height)
            {
                throw new ArgumentOutOfRangeException(
                    "region", "The ONNX image region must be fully inside the source image.");
            }
        }

        private static void ValidatePackedPixels(int columnPitch, int expectedPitch, string imageType)
        {
            if (columnPitch != expectedPitch)
            {
                throw new NotSupportedException(string.Format(
                    "ONNX {0} input requires packed {1}-byte pixels, but the source column pitch is {2}.",
                    imageType, expectedPitch, columnPitch));
            }
        }
    }
}
