using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Euresys.Open_eVision_22_12;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CSharpYolo461
{
    internal static class Program
    {
        private const int ImageSize = 224;
        private static readonly string[] ClassNames = { "flipped", "normal" };
        private static readonly HashSet<string> Extensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"
        };

        private static int Main(string[] args)
        {
            if (args.Length != 2 && args.Length != 6)
            {
                Console.Error.WriteLine("Usage: CSharpYolo461.exe <model.onnx> <test-directory> [roi-x roi-y roi-width roi-height]");
                return 2;
            }

            var modelPath = Path.GetFullPath(args[0]);
            var testDirectory = Path.GetFullPath(args[1]);
            if (!File.Exists(modelPath) || !Directory.Exists(testDirectory))
            {
                Console.Error.WriteLine("Model or test directory does not exist.");
                return 2;
            }

            var images = Directory.EnumerateFiles(testDirectory, "*", SearchOption.AllDirectories)
                .Where(path => Extensions.Contains(Path.GetExtension(path)))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            RoiPlacement roiPlacement;
            if (!TryParseRoi(args, out roiPlacement))
            {
                Console.Error.WriteLine("ROI values must be integers; width and height must be positive.");
                return 2;
            }

            using (var classifier = new Classifier(modelPath, roiPlacement))
            {
                Console.WriteLine("Runtime: ONNX Runtime {0}, .NET Framework {1}",
                    OrtEnv.Instance().GetVersionString(), Environment.Version);
                Console.WriteLine("Model input: {0}; test images: {1}",
                    classifier.InputName, images.Length);
                Console.WriteLine("Input contract: " + classifier.InputDescription);
                Console.WriteLine("Euresys input: " + classifier.EuresysInputDescription +
                    (roiPlacement == null
                        ? " with full image"
                        : string.Format(" with ROI ({0}, {1}, {2}, {3})",
                            roiPlacement.X, roiPlacement.Y, roiPlacement.Width, roiPlacement.Height)));

                for (var i = 0; i < Math.Min(20, images.Length); i++)
                    classifier.Predict(images[i]);
                classifier.ResetTimings();

                var stopwatch = Stopwatch.StartNew();
                var correct = 0;
                var flippedCorrect = 0;
                var flippedTotal = 0;
                var normalCorrect = 0;
                var normalTotal = 0;
                var errors = new List<string>();

                foreach (var image in images)
                {
                    var expected = new DirectoryInfo(Path.GetDirectoryName(image)).Name;
                    var prediction = classifier.Predict(image);
                    var isCorrect = string.Equals(expected, prediction.Name, StringComparison.OrdinalIgnoreCase);
                    if (isCorrect)
                        correct++;
                    else
                        errors.Add(string.Format("{0} -> {1} ({2:P2})", MakeRelative(testDirectory, image), prediction.Name, prediction.Confidence));

                    if (string.Equals(expected, "flipped", StringComparison.OrdinalIgnoreCase))
                    {
                        flippedTotal++;
                        if (isCorrect) flippedCorrect++;
                    }
                    else if (string.Equals(expected, "normal", StringComparison.OrdinalIgnoreCase))
                    {
                        normalTotal++;
                        if (isCorrect) normalCorrect++;
                    }
                }

                stopwatch.Stop();
                Console.WriteLine("Accuracy: {0}/{1} ({2:P2})", correct, images.Length, (double)correct / images.Length);
                Console.WriteLine("Flipped recall: {0}/{1} ({2:P2})", flippedCorrect, flippedTotal, (double)flippedCorrect / flippedTotal);
                Console.WriteLine("Normal recall: {0}/{1} ({2:P2})", normalCorrect, normalTotal, (double)normalCorrect / normalTotal);
                Console.WriteLine("End-to-end: {0:F3} ms/image ({1:F1} images/s)",
                    stopwatch.Elapsed.TotalMilliseconds / images.Length,
                    images.Length / stopwatch.Elapsed.TotalSeconds);
                Console.WriteLine("Preprocess: {0:F3} ms/image", classifier.PreprocessMilliseconds / images.Length);
                Console.WriteLine("{0}: {1:F3} ms/image",
                    classifier.HasEmbeddedPreprocessing ? "ONNX graph (preprocess + inference)" : "Inference",
                    classifier.InferenceMilliseconds / images.Length);
                foreach (var error in errors)
                    Console.WriteLine("Mismatch: " + error);
            }

            return 0;
        }

        private enum InputContract
        {
            FloatNchw,
            Bw8Nchw,
            C24Nhwc
        }

        private sealed class Classifier : IDisposable
        {
            private const int PlaneSize = ImageSize * ImageSize;
            private readonly InferenceSession session;
            private readonly EROIBW8 bw8Roi;
            private readonly EROIC24 c24Roi;
            private readonly RoiPlacement roiPlacement;
            private readonly InputContract inputContract;
            private readonly float[] tensorBuffer;
            private byte[] byteTensorBuffer;
            private int byteTensorWidth;
            private int byteTensorHeight;
            private IReadOnlyCollection<NamedOnnxValue> inputs;
            private long preprocessTicks;
            private long inferenceTicks;

            public Classifier(string modelPath, RoiPlacement roiPlacement)
            {
                using (var options = new SessionOptions())
                    session = new InferenceSession(modelPath, options);

                var inputMetadata = session.InputMetadata.Single();
                InputName = inputMetadata.Key;
                var dimensions = inputMetadata.Value.Dimensions;
                if (inputMetadata.Value.ElementType == typeof(float))
                    inputContract = InputContract.FloatNchw;
                else if (inputMetadata.Value.ElementType == typeof(byte) && dimensions.Length == 4 && dimensions[1] == 1)
                    inputContract = InputContract.Bw8Nchw;
                else if (inputMetadata.Value.ElementType == typeof(byte) && dimensions.Length == 4 && dimensions[3] == 3)
                    inputContract = InputContract.C24Nhwc;
                else
                    throw new NotSupportedException("Expected float NCHW, uint8 BW8 NCHW, or uint8 C24 NHWC input metadata.");

                if (inputContract == InputContract.FloatNchw)
                {
                    InputDescription = "float NCHW; resize and normalization run in C#";
                    EuresysInputDescription = "EImageBW8 / EROIBW8";
                }
                else if (inputContract == InputContract.Bw8Nchw)
                {
                    InputDescription = "uint8 [1,1,H,W] BW8 NCHW; preprocessing runs inside ONNX";
                    EuresysInputDescription = "EImageBW8 / EROIBW8";
                }
                else
                {
                    InputDescription = "uint8 [1,H,W,3] C24 NHWC BGR; channel reorder and preprocessing run inside ONNX";
                    EuresysInputDescription = "EImageC24 / EROIC24";
                }
                this.roiPlacement = roiPlacement;
                if (inputContract == InputContract.C24Nhwc)
                    c24Roi = new EROIC24();
                else
                    bw8Roi = new EROIBW8();

                if (inputContract == InputContract.FloatNchw)
                {
                    tensorBuffer = new float[3 * PlaneSize];
                    var tensor = new DenseTensor<float>(tensorBuffer, new[] { 1, 3, ImageSize, ImageSize });
                    inputs = new[] { NamedOnnxValue.CreateFromTensor(InputName, tensor) };
                }
            }

            public string InputName { get; private set; }
            public string InputDescription { get; private set; }
            public string EuresysInputDescription { get; private set; }
            public bool HasEmbeddedPreprocessing { get { return inputContract != InputContract.FloatNchw; } }
            public double PreprocessMilliseconds { get { return preprocessTicks * 1000.0 / Stopwatch.Frequency; } }
            public double InferenceMilliseconds { get { return inferenceTicks * 1000.0 / Stopwatch.Frequency; } }

            public Prediction Predict(string imagePath)
            {
                var started = Stopwatch.GetTimestamp();
                if (inputContract == InputContract.C24Nhwc)
                {
                    using (var image = new EImageC24())
                    {
                        image.Load(imagePath);
                        return Predict(image, started);
                    }
                }
                else
                {
                    using (var image = new EImageBW8())
                    {
                        image.Load(imagePath);
                        return Predict(image, started);
                    }
                }
            }

            // Production entry point when acquisition/vision processing already owns the Euresys image.
            public Prediction Predict(EImageBW8 image)
            {
                if (inputContract == InputContract.C24Nhwc)
                    throw new InvalidOperationException("This model requires EImageC24 input.");
                return Predict(image, Stopwatch.GetTimestamp());
            }

            // Production entry point for an Open eVision C24 image or ROI source.
            public Prediction Predict(EImageC24 image)
            {
                if (inputContract != InputContract.C24Nhwc)
                    throw new InvalidOperationException("This model requires EImageBW8 input.");
                return Predict(image, Stopwatch.GetTimestamp());
            }

            private Prediction Predict(EImageBW8 image, long started)
            {
                if (image == null)
                    throw new ArgumentNullException("image");

                LoadTensor(image);
                return RunInference(started);
            }

            private Prediction Predict(EImageC24 image, long started)
            {
                if (image == null)
                    throw new ArgumentNullException("image");

                LoadTensor(image);
                return RunInference(started);
            }

            private Prediction RunInference(long started)
            {
                var preprocessed = Stopwatch.GetTimestamp();
                Prediction prediction;
                using (var results = session.Run(inputs))
                {
                    var scores = results.First().AsTensor<float>();
                    var first = scores.GetValue(0);
                    var second = scores.GetValue(1);
                    var bestIndex = second > first ? 1 : 0;
                    prediction = new Prediction(ClassNames[bestIndex], SoftmaxConfidence(first, second, bestIndex));
                }
                var inferred = Stopwatch.GetTimestamp();
                preprocessTicks += preprocessed - started;
                inferenceTicks += inferred - preprocessed;
                return prediction;
            }

            public void ResetTimings()
            {
                preprocessTicks = 0;
                inferenceTicks = 0;
            }

            public void Dispose()
            {
                if (bw8Roi != null)
                    bw8Roi.Dispose();
                if (c24Roi != null)
                    c24Roi.Dispose();
                session.Dispose();
            }

            private unsafe void LoadTensor(EImageBW8 image)
            {
                AttachBw8Roi(image);
                try
                {
                    if (inputContract == InputContract.Bw8Nchw)
                    {
                        LoadBw8TensorFromRoi();
                        return;
                    }

                    for (var y = 0; y < ImageSize; y++)
                    {
                        var sourceY = ((y + 0.5) * bw8Roi.Height / ImageSize) - 0.5;
                        var y0 = Math.Max(0, Math.Min(bw8Roi.Height - 1, (int)Math.Floor(sourceY)));
                        var y1 = Math.Min(y0 + 1, bw8Roi.Height - 1);
                        var yWeight = sourceY <= 0 ? 0.0 : sourceY - Math.Floor(sourceY);
                        var row0 = (byte*)bw8Roi.GetImagePtr(0, y0).ToPointer();
                        var row1 = (byte*)bw8Roi.GetImagePtr(0, y1).ToPointer();
                        var tensorRow = y * ImageSize;
                        for (var x = 0; x < ImageSize; x++)
                        {
                            var sourceX = ((x + 0.5) * bw8Roi.Width / ImageSize) - 0.5;
                            var x0 = Math.Max(0, Math.Min(bw8Roi.Width - 1, (int)Math.Floor(sourceX)));
                            var x1 = Math.Min(x0 + 1, bw8Roi.Width - 1);
                            var xWeight = sourceX <= 0 ? 0.0 : sourceX - Math.Floor(sourceX);
                            var top = row0[x0] + (row0[x1] - row0[x0]) * xWeight;
                            var bottom = row1[x0] + (row1[x1] - row1[x0]) * xWeight;
                            var pixel = (float)((top + (bottom - top) * yWeight) / 255.0);
                            var tensorOffset = tensorRow + x;
                            tensorBuffer[tensorOffset] = pixel;
                            tensorBuffer[PlaneSize + tensorOffset] = pixel;
                            tensorBuffer[2 * PlaneSize + tensorOffset] = pixel;
                        }
                    }
                }
                finally
                {
                    bw8Roi.Detach();
                }
            }

            private void LoadTensor(EImageC24 image)
            {
                AttachC24Roi(image);
                try
                {
                    LoadC24TensorFromRoi();
                }
                finally
                {
                    c24Roi.Detach();
                }
            }

            private void AttachBw8Roi(EImageBW8 image)
            {
                var placement = roiPlacement ?? new RoiPlacement(0, 0, image.Width, image.Height);
                ValidatePlacement(placement, image.Width, image.Height);

                bw8Roi.Detach();
                bw8Roi.Attach(image);
                bw8Roi.SetPlacement(placement.X, placement.Y, placement.Width, placement.Height);
            }

            private void AttachC24Roi(EImageC24 image)
            {
                var placement = roiPlacement ?? new RoiPlacement(0, 0, image.Width, image.Height);
                ValidatePlacement(placement, image.Width, image.Height);

                c24Roi.Detach();
                c24Roi.Attach(image);
                c24Roi.SetPlacement(placement.X, placement.Y, placement.Width, placement.Height);
            }

            private static void ValidatePlacement(RoiPlacement placement, int imageWidth, int imageHeight)
            {
                if (placement.X < 0 || placement.Y < 0 ||
                    placement.X + placement.Width > imageWidth ||
                    placement.Y + placement.Height > imageHeight)
                    throw new ArgumentOutOfRangeException("image",
                        string.Format("ROI ({0}, {1}, {2}, {3}) is outside image {4}x{5}.",
                            placement.X, placement.Y, placement.Width, placement.Height, imageWidth, imageHeight));
            }

            private void LoadBw8TensorFromRoi()
            {
                var width = bw8Roi.Width;
                var height = bw8Roi.Height;
                EnsureByteTensor(width, height, 1);

                for (var y = 0; y < height; y++)
                    Marshal.Copy(bw8Roi.GetImagePtr(0, y), byteTensorBuffer, y * width, width);
            }

            private void LoadC24TensorFromRoi()
            {
                var width = c24Roi.Width;
                var height = c24Roi.Height;
                var rowBytes = checked(width * 3);
                EnsureByteTensor(width, height, 3);

                for (var y = 0; y < height; y++)
                    Marshal.Copy(c24Roi.GetImagePtr(0, y), byteTensorBuffer, y * rowBytes, rowBytes);
            }

            private void EnsureByteTensor(int width, int height, int channels)
            {
                var requiredLength = checked(width * height * channels);
                if (byteTensorBuffer != null && byteTensorBuffer.Length == requiredLength &&
                    byteTensorWidth == width && byteTensorHeight == height)
                    return;

                byteTensorBuffer = new byte[requiredLength];
                byteTensorWidth = width;
                byteTensorHeight = height;
                var shape = inputContract == InputContract.Bw8Nchw
                    ? new[] { 1, 1, height, width }
                    : new[] { 1, height, width, 3 };
                var tensor = new DenseTensor<byte>(byteTensorBuffer, shape);
                inputs = new[] { NamedOnnxValue.CreateFromTensor(InputName, tensor) };
            }
        }

        private static bool TryParseRoi(string[] args, out RoiPlacement placement)
        {
            placement = null;
            if (args.Length == 2)
                return true;

            int x;
            int y;
            int width;
            int height;
            if (!int.TryParse(args[2], out x) || !int.TryParse(args[3], out y) ||
                !int.TryParse(args[4], out width) || !int.TryParse(args[5], out height) ||
                width <= 0 || height <= 0)
                return false;

            placement = new RoiPlacement(x, y, width, height);
            return true;
        }

        private sealed class RoiPlacement
        {
            public RoiPlacement(int x, int y, int width, int height)
            {
                X = x;
                Y = y;
                Width = width;
                Height = height;
            }

            public int X { get; private set; }
            public int Y { get; private set; }
            public int Width { get; private set; }
            public int Height { get; private set; }
        }

        private static float SoftmaxConfidence(float first, float second, int index)
        {
            var maximum = Math.Max(first, second);
            var firstExp = Math.Exp(first - maximum);
            var secondExp = Math.Exp(second - maximum);
            return (float)((index == 0 ? firstExp : secondExp) / (firstExp + secondExp));
        }

        private static string MakeRelative(string root, string path)
        {
            var rootUri = new Uri(root.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar);
            return Uri.UnescapeDataString(rootUri.MakeRelativeUri(new Uri(path)).ToString()).Replace('/', Path.DirectorySeparatorChar);
        }

        private sealed class Prediction
        {
            public Prediction(string name, float confidence)
            {
                Name = name;
                Confidence = confidence;
            }

            public string Name { get; private set; }
            public float Confidence { get; private set; }
        }
    }
}
