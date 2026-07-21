using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;

namespace CSharpYolo461
{
    internal static class Program
    {
        private static readonly HashSet<string> Extensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"
        };

        private static int Main(string[] args)
        {
            if (args.Length != 2 && args.Length != 3 && args.Length != 6 && args.Length != 7)
            {
                Console.Error.WriteLine("Usage: CSharpYolo461.exe <model.onnx> <test-directory> [cpu|directml] [roi-x roi-y roi-width roi-height]");
                return 2;
            }

            var modelPath = Path.GetFullPath(args[0]);
            var testDirectory = Path.GetFullPath(args[1]);
            if (!File.Exists(modelPath) || !Directory.Exists(testDirectory))
            {
                Console.Error.WriteLine("Model or test directory does not exist.");
                return 2;
            }

            RoiPlacement roiPlacement;
            if (!TryParseRoi(args, out roiPlacement))
            {
                Console.Error.WriteLine("ROI values must be integers; width and height must be positive.");
                return 2;
            }

            ExecutionProvider executionProvider;
            if (!TryParseExecutionProvider(args, out executionProvider))
            {
                Console.Error.WriteLine("Execution provider must be 'cpu' or 'directml'.");
                return 2;
            }

            var images = Directory.EnumerateFiles(testDirectory, "*", SearchOption.AllDirectories)
                .Where(path => Extensions.Contains(Path.GetExtension(path)))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            using (var classifier = new YoloClassifier(modelPath, new[] { "flipped", "normal" }, roiPlacement, executionProvider))
            {
                PrintModelInformation(classifier, images.Length, roiPlacement);

                for (var i = 0; i < Math.Min(20, images.Length); i++)
                    classifier.Predict(images[i]);
                classifier.ResetTimings();

                RunEvaluation(classifier, testDirectory, images);
            }

            return 0;
        }

        private static void PrintModelInformation(YoloClassifier classifier, int imageCount, RoiPlacement roiPlacement)
        {
            Console.WriteLine("Runtime: ONNX Runtime {0}, .NET Framework {1}",
                OrtEnv.Instance().GetVersionString(), Environment.Version);
            Console.WriteLine("Execution provider: {0}", classifier.ExecutionProvider);
            Console.WriteLine("Model input: {0}; test images: {1}", classifier.InputName, imageCount);
            Console.WriteLine("Input contract: " + classifier.InputDescription);
            Console.WriteLine("Euresys input: " + classifier.EuresysInputDescription +
                (roiPlacement == null
                    ? " with full image"
                    : string.Format(" with ROI ({0}, {1}, {2}, {3})",
                        roiPlacement.X, roiPlacement.Y, roiPlacement.Width, roiPlacement.Height)));
        }

        private static void RunEvaluation(YoloClassifier classifier, string testDirectory, string[] images)
        {
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
            Console.WriteLine("Accuracy: {0}/{1} ({2:P2})", correct, images.Length, Divide(correct, images.Length));
            Console.WriteLine("Flipped recall: {0}/{1} ({2:P2})", flippedCorrect, flippedTotal, Divide(flippedCorrect, flippedTotal));
            Console.WriteLine("Normal recall: {0}/{1} ({2:P2})", normalCorrect, normalTotal, Divide(normalCorrect, normalTotal));
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

        private static double Divide(int numerator, int denominator)
        {
            return denominator == 0 ? 0.0 : (double)numerator / denominator;
        }

        private static bool TryParseRoi(string[] args, out RoiPlacement placement)
        {
            placement = null;
            if (args.Length == 2 || args.Length == 3)
                return true;

            var offset = args.Length == 7 ? 3 : 2;
            int x;
            int y;
            int width;
            int height;
            if (!int.TryParse(args[offset], out x) || !int.TryParse(args[offset + 1], out y) ||
                !int.TryParse(args[offset + 2], out width) || !int.TryParse(args[offset + 3], out height) ||
                width <= 0 || height <= 0)
                return false;

            placement = new RoiPlacement(x, y, width, height);
            return true;
        }

        private static bool TryParseExecutionProvider(string[] args, out ExecutionProvider executionProvider)
        {
            executionProvider = ExecutionProvider.Cpu;
            if (args.Length == 2 || args.Length == 6)
                return true;

            if (string.Equals(args[2], "cpu", StringComparison.OrdinalIgnoreCase))
                return true;
            if (string.Equals(args[2], "directml", StringComparison.OrdinalIgnoreCase))
            {
                executionProvider = ExecutionProvider.DirectML;
                return true;
            }

            return false;
        }

        private static string MakeRelative(string root, string path)
        {
            var rootUri = new Uri(root.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar);
            return Uri.UnescapeDataString(rootUri.MakeRelativeUri(new Uri(path)).ToString()).Replace('/', Path.DirectorySeparatorChar);
        }
    }
}
