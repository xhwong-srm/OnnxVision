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

            using (var options = new SessionOptions())
            using (var session = new InferenceSession(modelPath, options))
            {
                Console.WriteLine("Runtime: ONNX Runtime {0}, .NET Framework {1}",
                    OrtEnv.Instance().GetVersionString(), Environment.Version);
                Console.WriteLine("Model input: {0}; test images: {1}",
                    session.InputMetadata.Single().Key, images.Length);

                for (var i = 0; i < Math.Min(20, images.Length); i++)
                    Predict(session, images[i]);

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
                    var prediction = Predict(session, image);
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
                foreach (var error in errors)
                    Console.WriteLine("Mismatch: " + error);
            }

            return 0;
        }

        private static Prediction Predict(InferenceSession session, string imagePath)
        {
            var tensor = LoadTensor(imagePath);
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("images", tensor) };
            using (var results = session.Run(inputs))
            {
                var scores = results.First().AsEnumerable<float>().ToArray();
                var bestIndex = scores[1] > scores[0] ? 1 : 0;
                return new Prediction(ClassNames[bestIndex], SoftmaxConfidence(scores, bestIndex));
            }
        }

        private static DenseTensor<float> LoadTensor(string imagePath)
        {
            using (var source = new Bitmap(imagePath))
            using (var resized = new Bitmap(ImageSize, ImageSize, PixelFormat.Format24bppRgb))
            {
                using (var graphics = Graphics.FromImage(resized))
                {
                    graphics.CompositingMode = CompositingMode.SourceCopy;
                    graphics.InterpolationMode = InterpolationMode.HighQualityBilinear;
                    graphics.PixelOffsetMode = PixelOffsetMode.Half;
                    graphics.DrawImage(source, new Rectangle(0, 0, ImageSize, ImageSize), 0, 0, source.Width, source.Height, GraphicsUnit.Pixel);
                }

                var tensor = new DenseTensor<float>(new[] { 1, 3, ImageSize, ImageSize });
                var rectangle = new Rectangle(0, 0, ImageSize, ImageSize);
                var data = resized.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                try
                {
                    var bytes = new byte[Math.Abs(data.Stride) * ImageSize];
                    System.Runtime.InteropServices.Marshal.Copy(data.Scan0, bytes, 0, bytes.Length);
                    for (var y = 0; y < ImageSize; y++)
                    {
                        var row = data.Stride > 0 ? y * data.Stride : (ImageSize - 1 - y) * -data.Stride;
                        for (var x = 0; x < ImageSize; x++)
                        {
                            var offset = row + x * 3;
                            tensor[0, 0, y, x] = bytes[offset + 2] / 255f;
                            tensor[0, 1, y, x] = bytes[offset + 1] / 255f;
                            tensor[0, 2, y, x] = bytes[offset] / 255f;
                        }
                    }
                }
                finally
                {
                    resized.UnlockBits(data);
                }
                return tensor;
            }
        }

        private static float SoftmaxConfidence(float[] scores, int index)
        {
            var maximum = scores.Max();
            var denominator = scores.Sum(score => Math.Exp(score - maximum));
            return (float)(Math.Exp(scores[index] - maximum) / denominator);
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
