using System;
using System.Collections.Generic;
using System.Linq;
using Euresys.Open_eVision_22_12;
using OnnxVision.Classification;
using OnnxVision.Detection;
using OnnxVision.Euresys;
using OnnxVision.Imaging;

namespace OnnxVision
{
    internal static partial class Program
    {
        private static List<LoadedImage> LoadImages(string[] paths, bool color)
        {
            var images = new List<LoadedImage>(paths.Length);
            try
            {
                foreach (string path in paths)
                    images.Add(LoadedImage.Load(path, color));
                return images;
            }
            catch
            {
                DisposeImages(images);
                throw;
            }
        }

        private static void DisposeImages(IEnumerable<LoadedImage> images)
        {
            if (images == null)
                return;
            foreach (LoadedImage image in images)
                image.Dispose();
        }

        private static List<LoadedImageBatch> BuildInferenceBatches(
            IReadOnlyList<LoadedImage> images, int batchSize, bool padFinalBatch)
        {
            var batches = new List<LoadedImageBatch>();
            for (int offset = 0; offset < images.Count; offset += batchSize)
            {
                int logicalCount = Math.Min(batchSize, images.Count - offset);
                int physicalCount = padFinalBatch ? batchSize : logicalCount;
                var physicalImages = new List<LoadedImage>(physicalCount);
                for (int index = 0; index < logicalCount; index++)
                    physicalImages.Add(images[offset + index]);
                while (physicalImages.Count < physicalCount)
                    physicalImages.Add(physicalImages[physicalImages.Count - 1]);
                batches.Add(new LoadedImageBatch(physicalImages, logicalCount));
            }
            return batches;
        }

        private sealed class LoadedImageBatch
        {
            public LoadedImageBatch(IReadOnlyList<LoadedImage> images, int logicalCount)
            {
                Images = images;
                LogicalCount = logicalCount;
            }

            public IReadOnlyList<LoadedImage> Images { get; private set; }
            public int LogicalCount { get; private set; }

            public IReadOnlyList<OnnxClassification> Classify(
                OnnxClassificationModel model, RoiPlacement roi)
            {
                if (Images.Count == 1)
                    return new[] { Images[0].Classify(model, roi) };

                return model.ClassifyBatch(Images.Select(image => image.CreateBuffer(roi)).ToArray());
            }

            public IReadOnlyList<IReadOnlyList<OnnxDetection>> Detect(
                OnnxObjectDetectionModel model, float confidenceThreshold, float nmsIouThreshold)
            {
                return model.DetectBatch(Images.Select(image => image.CreateBuffer(null)).ToArray(),
                    confidenceThreshold, nmsIouThreshold);
            }
        }

        private sealed class LoadedImage : IDisposable
        {
            private LoadedImage(string path, EImageBW8 bw8, EImageC24 c24)
            {
                Path = path;
                Bw8 = bw8;
                C24 = c24;
            }

            public string Path { get; private set; }
            private EImageBW8 Bw8 { get; set; }
            private EImageC24 C24 { get; set; }

            public static LoadedImage Load(string path, bool color)
            {
                if (color)
                {
                    var image = new EImageC24();
                    try
                    {
                        image.Load(path);
                        return new LoadedImage(path, null, image);
                    }
                    catch
                    {
                        image.Dispose();
                        throw;
                    }
                }

                var bw8 = new EImageBW8();
                try
                {
                    bw8.Load(path);
                    return new LoadedImage(path, bw8, null);
                }
                catch
                {
                    bw8.Dispose();
                    throw;
                }
            }

            public OnnxClassification Classify(OnnxClassificationModel model, RoiPlacement roi)
            {
                if (C24 != null)
                    return roi == null ? model.Classify(C24) : model.Classify(C24, roi.ToRectangle());
                return roi == null ? model.Classify(Bw8) : model.Classify(Bw8, roi.ToRectangle());
            }

            public OnnxImageBuffer CreateBuffer(RoiPlacement roi)
            {
                if (C24 != null)
                {
                    return roi == null
                        ? EuresysOnnxExtensions.CreateImageBuffer(C24,
                            new System.Drawing.Rectangle(0, 0, C24.Width, C24.Height))
                        : EuresysOnnxExtensions.CreateImageBuffer(C24, roi.ToRectangle());
                }
                return roi == null
                    ? EuresysOnnxExtensions.CreateImageBuffer(Bw8,
                        new System.Drawing.Rectangle(0, 0, Bw8.Width, Bw8.Height))
                    : EuresysOnnxExtensions.CreateImageBuffer(Bw8, roi.ToRectangle());
            }

            public IReadOnlyList<OnnxDetection> Detect(
                OnnxObjectDetectionModel model, float confidenceThreshold, float nmsIouThreshold)
            {
                return C24 != null
                    ? model.Detect(C24, confidenceThreshold, nmsIouThreshold)
                    : model.Detect(Bw8, confidenceThreshold, nmsIouThreshold);
            }

            public void Dispose()
            {
                if (Bw8 != null)
                {
                    Bw8.Dispose();
                    Bw8 = null;
                }
                if (C24 != null)
                {
                    C24.Dispose();
                    C24 = null;
                }
            }
        }
    }
}
