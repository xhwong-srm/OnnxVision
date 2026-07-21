using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
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
            if (args.Length != 2)
            {
                Console.Error.WriteLine("Usage: CSharpYolo461.exe <model.onnx> <test-directory>");
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

            using (var classifier = new Classifier(modelPath))
            {
                Console.WriteLine("Runtime: ONNX Runtime {0}, .NET Framework {1}",
                    OrtEnv.Instance().GetVersionString(), Environment.Version);
                Console.WriteLine("Model input: {0}; test images: {1}",
                    classifier.InputName, images.Length);

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
                Console.WriteLine("Inference: {0:F3} ms/image", classifier.InferenceMilliseconds / images.Length);
                foreach (var error in errors)
                    Console.WriteLine("Mismatch: " + error);
            }

            return 0;
        }

        private sealed class Classifier : IDisposable
        {
            private const int PlaneSize = ImageSize * ImageSize;
            private readonly InferenceSession session;
            private readonly Bitmap resized;
            private readonly Graphics graphics;
            private readonly float[] tensorBuffer;
            private readonly IReadOnlyCollection<NamedOnnxValue> inputs;
            private long preprocessTicks;
            private long inferenceTicks;

            public Classifier(string modelPath)
            {
                using (var options = new SessionOptions())
                    session = new InferenceSession(modelPath, options);

                InputName = session.InputMetadata.Single().Key;
                resized = new Bitmap(ImageSize, ImageSize, PixelFormat.Format24bppRgb);
                graphics = Graphics.FromImage(resized);
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.InterpolationMode = InterpolationMode.HighQualityBilinear;
                graphics.PixelOffsetMode = PixelOffsetMode.Half;

                tensorBuffer = new float[3 * PlaneSize];
                var tensor = new DenseTensor<float>(tensorBuffer, new[] { 1, 3, ImageSize, ImageSize });
                inputs = new[] { NamedOnnxValue.CreateFromTensor(InputName, tensor) };
            }

            public string InputName { get; private set; }
            public double PreprocessMilliseconds { get { return preprocessTicks * 1000.0 / Stopwatch.Frequency; } }
            public double InferenceMilliseconds { get { return inferenceTicks * 1000.0 / Stopwatch.Frequency; } }

            public Prediction Predict(string imagePath)
            {
                var started = Stopwatch.GetTimestamp();
                LoadTensor(imagePath);
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
                graphics.Dispose();
                resized.Dispose();
                session.Dispose();
            }

            private unsafe void LoadTensor(string imagePath)
            {
                using (var source = new Bitmap(imagePath))
                    graphics.DrawImage(source, new Rectangle(0, 0, ImageSize, ImageSize), 0, 0, source.Width, source.Height, GraphicsUnit.Pixel);

                var rectangle = new Rectangle(0, 0, ImageSize, ImageSize);
                var data = resized.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                try
                {
                    for (var y = 0; y < ImageSize; y++)
                    {
                        var row = (byte*)data.Scan0 + y * data.Stride;
                        var tensorRow = y * ImageSize;
                        for (var x = 0; x < ImageSize; x++)
                        {
                            var sourceOffset = x * 3;
                            var tensorOffset = tensorRow + x;
                            tensorBuffer[tensorOffset] = row[sourceOffset + 2] / 255f;
                            tensorBuffer[PlaneSize + tensorOffset] = row[sourceOffset + 1] / 255f;
                            tensorBuffer[2 * PlaneSize + tensorOffset] = row[sourceOffset] / 255f;
                        }
                    }
                }
                finally
                {
                    resized.UnlockBits(data);
                }
            }
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
