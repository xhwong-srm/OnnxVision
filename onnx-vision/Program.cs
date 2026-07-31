using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;

namespace OnnxVision
{
    internal static class Program
    {
        private static readonly HashSet<string> Extensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"
        };

        private static int Main(string[] args)
        {
            if (args.Length > 0 && string.Equals(args[0], "benchmark-detect", StringComparison.OrdinalIgnoreCase))
                return RunDetectionBenchmark(args);
            if (args.Length > 0 && string.Equals(args[0], "detect", StringComparison.OrdinalIgnoreCase))
                return RunDetection(args);

            if (args.Length != 2 && args.Length != 3 && args.Length != 6 && args.Length != 7)
            {
                Console.Error.WriteLine("Usage: OnnxVision.exe <model.onnx> <test-directory> [cpu|directml|openvino-cpu|openvino-gpu] [roi-x roi-y roi-width roi-height]");
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
                Console.Error.WriteLine("Execution provider must be 'cpu', 'directml', 'openvino-cpu', or 'openvino-gpu'.");
                return 2;
            }

            var images = Directory.EnumerateFiles(testDirectory, "*", SearchOption.AllDirectories)
                .Where(path => Extensions.Contains(Path.GetExtension(path)))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            using (var classifier = new ImageClassifier(modelPath, null, roiPlacement, executionProvider))
            {
                PrintModelInformation(classifier, images.Length, roiPlacement);

                for (var i = 0; i < Math.Min(20, images.Length); i++)
                    classifier.Predict(images[i]);
                classifier.ResetTimings();

                RunEvaluation(classifier, testDirectory, images);
            }

            return 0;
        }

        private static int RunDetectionBenchmark(string[] args)
        {
            if (args.Length < 3 || args.Length > 6)
            {
                Console.Error.WriteLine(
                    "Usage: OnnxVision.exe benchmark-detect <model.onnx> <image-directory> [confidence] [repeats] [cpu|directml|openvino-cpu|openvino-gpu]");
                return 2;
            }

            var modelPath = Path.GetFullPath(args[1]);
            var imageDirectory = Path.GetFullPath(args[2]);
            var threshold = 0.5f;
            var repeats = 3;
            var provider = ExecutionProvider.Cpu;
            if (args.Length >= 4 && !float.TryParse(
                args[3], System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out threshold))
            {
                Console.Error.WriteLine("Confidence must be a number between zero and one.");
                return 2;
            }
            if (args.Length >= 5 && (!int.TryParse(args[4], out repeats) || repeats < 1))
            {
                Console.Error.WriteLine("Repeats must be a positive integer.");
                return 2;
            }
            if (args.Length == 6 && !TryParseExecutionProviderName(args[5], out provider))
            {
                Console.Error.WriteLine("Execution provider must be 'cpu', 'directml', 'openvino-cpu', or 'openvino-gpu'.");
                return 2;
            }
            if (!File.Exists(modelPath) || !Directory.Exists(imageDirectory))
            {
                Console.Error.WriteLine("Model or image directory does not exist.");
                return 2;
            }

            var images = Directory.EnumerateFiles(imageDirectory, "*", SearchOption.AllDirectories)
                .Where(path => Extensions.Contains(Path.GetExtension(path)))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            if (images.Length == 0)
            {
                Console.Error.WriteLine("No supported images were found.");
                return 2;
            }

            var construction = Stopwatch.StartNew();
            using (var detector = new ObjectDetector(modelPath, null, provider))
            {
                construction.Stop();
                var warmupRuns = Math.Min(10, images.Length);
                for (var index = 0; index < warmupRuns; index++)
                    detector.Detect(images[index], threshold, 0.7f);
                detector.ResetTimings();

                var classCounts = detector.ClassNames.ToDictionary(name => name, name => 0);
                var confidenceSum = 0.0;
                var detectionCount = 0;
                var elapsed = Stopwatch.StartNew();
                for (var repeat = 0; repeat < repeats; repeat++)
                {
                    foreach (var image in images)
                    {
                        var detections = detector.Detect(image, threshold, 0.7f);
                        detectionCount += detections.Count;
                        foreach (var detection in detections)
                        {
                            classCounts[detection.Name]++;
                            confidenceSum += detection.Confidence;
                        }
                    }
                }
                elapsed.Stop();

                var executions = checked(images.Length * repeats);
                Console.WriteLine("Provider: {0}", provider);
                Console.WriteLine("Images: {0}; repeats: {1}; executions: {2}; warmups: {3}",
                    images.Length, repeats, executions, warmupRuns);
                Console.WriteLine("Session construction: {0:F3} ms", construction.Elapsed.TotalMilliseconds);
                Console.WriteLine("Wall time: {0:F3} ms/image ({1:F2} images/s)",
                    elapsed.Elapsed.TotalMilliseconds / executions,
                    executions / elapsed.Elapsed.TotalSeconds);
                Console.WriteLine("Preprocess: {0:F3} ms/image", detector.PreprocessMilliseconds / executions);
                Console.WriteLine("ONNX inference: {0:F3} ms/image", detector.InferenceMilliseconds / executions);
                Console.WriteLine("Detections: {0}; confidence sum: {1:F6}", detectionCount, confidenceSum);
                Console.WriteLine("Class counts: " + string.Join(", ",
                    classCounts.Select(item => item.Key + "=" + item.Value)));
            }
            return 0;
        }

        private static int RunDetection(string[] args)
        {
            if (args.Length < 3 || args.Length > 5)
            {
                Console.Error.WriteLine(
                    "Usage: OnnxVision.exe detect <model.onnx> <image-or-directory> [confidence] [cpu|directml|openvino-cpu|openvino-gpu]");
                return 2;
            }

            var modelPath = Path.GetFullPath(args[1]);
            var imageSource = Path.GetFullPath(args[2]);
            var threshold = 0.5f;
            if (args.Length >= 4 && !float.TryParse(
                args[3], System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out threshold))
            {
                Console.Error.WriteLine("Confidence must be a number between zero and one.");
                return 2;
            }
            var provider = ExecutionProvider.Cpu;
            if (args.Length == 5 && !TryParseExecutionProviderName(args[4], out provider))
            {
                Console.Error.WriteLine("Execution provider must be 'cpu', 'directml', 'openvino-cpu', or 'openvino-gpu'.");
                return 2;
            }
            if (!File.Exists(modelPath) || (!File.Exists(imageSource) && !Directory.Exists(imageSource)))
            {
                Console.Error.WriteLine("Model or image source does not exist.");
                return 2;
            }
            var images = File.Exists(imageSource)
                ? new[] { imageSource }
                : Directory.EnumerateFiles(imageSource, "*", SearchOption.AllDirectories)
                    .Where(path => Extensions.Contains(Path.GetExtension(path)))
                    .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                    .ToArray();

            using (var detector = new ObjectDetector(modelPath, null, provider))
            {
                Console.WriteLine("Detection contract: onnx-vision-detection-v1");
                Console.WriteLine("Input contract: " + detector.InputDescription);
                foreach (var image in images)
                {
                    var detections = detector.Detect(image, threshold, 0.7f);
                    Console.WriteLine("{0}: {1} detection(s)", image, detections.Count);
                    foreach (var detection in detections)
                        Console.WriteLine("  {0} {1:F4} [{2:F1}, {3:F1}, {4:F1}, {5:F1}]",
                            detection.Name, detection.Confidence,
                            detection.X1, detection.Y1, detection.X2, detection.Y2);
                }
                Console.WriteLine("Preprocess: {0:F3} ms/image", detector.PreprocessMilliseconds / Math.Max(1, images.Length));
                Console.WriteLine("ONNX inference: {0:F3} ms/image", detector.InferenceMilliseconds / Math.Max(1, images.Length));
            }
            return 0;
        }

        private static void PrintModelInformation(ImageClassifier classifier, int imageCount, RoiPlacement roiPlacement)
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

        private static void RunEvaluation(ImageClassifier classifier, string testDirectory, string[] images)
        {
            var stopwatch = Stopwatch.StartNew();
            var correct = 0;
            var flippedCorrect = 0;
            var flippedTotal = 0;
            var normalCorrect = 0;
            var normalTotal = 0;
            var truePositives = 0;
            var falsePositives = 0;
            var falseNegatives = 0;
            var trueNegatives = 0;
            var rocScores = new List<RocPoint>();
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

                var actualPositive = string.Equals(expected, "flipped", StringComparison.OrdinalIgnoreCase);
                var predictedPositive = string.Equals(prediction.Name, "flipped", StringComparison.OrdinalIgnoreCase);
                if (actualPositive && predictedPositive) truePositives++;
                else if (!actualPositive && predictedPositive) falsePositives++;
                else if (actualPositive) falseNegatives++;
                else trueNegatives++;

                var positiveIndex = predictedPositive ? prediction.ClassIndex : 1 - prediction.ClassIndex;
                if (prediction.Probabilities != null && prediction.Probabilities.Count > positiveIndex)
                    rocScores.Add(new RocPoint(actualPositive, prediction.Probabilities[positiveIndex]));
            }

            stopwatch.Stop();
            Console.WriteLine("Accuracy: {0}/{1} ({2:P2})", correct, images.Length, Divide(correct, images.Length));
            Console.WriteLine("Flipped recall: {0}/{1} ({2:P2})", flippedCorrect, flippedTotal, Divide(flippedCorrect, flippedTotal));
            Console.WriteLine("Normal recall: {0}/{1} ({2:P2})", normalCorrect, normalTotal, Divide(normalCorrect, normalTotal));
            PrintClassificationMetrics(truePositives, falsePositives, falseNegatives, trueNegatives, rocScores);
            PrintConfidenceInterval("Accuracy 95% Wilson CI", correct, images.Length, 1.96);
            PrintConfidenceInterval("Accuracy 99% Wilson CI", correct, images.Length, 2.576);
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

        private static void PrintClassificationMetrics(int tp, int fp, int fn, int tn, List<RocPoint> rocScores)
        {
            var precision = Divide(tp, tp + fp);
            var recall = Divide(tp, tp + fn);
            var f1 = Divide(2.0 * precision * recall, precision + recall);
            var normalPrecision = Divide(tn, tn + fn);
            var normalRecall = Divide(tn, tn + fp);
            var normalF1 = Divide(2.0 * normalPrecision * normalRecall, normalPrecision + normalRecall);
            Console.WriteLine("Confusion matrix (actual rows / predicted columns):");
            Console.WriteLine("                 flipped  normal");
            Console.WriteLine("actual flipped    {0,7} {1,7}", tp, fn);
            Console.WriteLine("actual normal     {0,7} {1,7}", fp, tn);
            Console.WriteLine("Flipped precision: {0:P2}; recall: {1:P2}; F1: {2:P2}", precision, recall, f1);
            Console.WriteLine("Normal precision: {0:P2}; recall: {1:P2}; F1: {2:P2}", normalPrecision, normalRecall, normalF1);
            Console.WriteLine("Macro precision: {0:P2}; macro recall: {1:P2}; macro F1: {2:P2}",
                (precision + normalPrecision) / 2.0,
                (recall + normalRecall) / 2.0,
                (f1 + normalF1) / 2.0);
            Console.WriteLine("ROC AUC (flipped positive): {0:F4}", CalculateAuc(rocScores));
        }

        private static void PrintConfidenceInterval(string label, int successes, int total, double z)
        {
            if (total == 0)
            {
                Console.WriteLine(label + ": n/a");
                return;
            }

            var p = (double)successes / total;
            var zSquared = z * z;
            var denominator = 1.0 + zSquared / total;
            var centre = p + zSquared / (2.0 * total);
            var margin = z * Math.Sqrt((p * (1.0 - p) + zSquared / (4.0 * total)) / total);
            Console.WriteLine("{0}: [{1:P2}, {2:P2}]", label,
                Math.Max(0.0, (centre - margin) / denominator),
                Math.Min(1.0, (centre + margin) / denominator));
        }

        private static double CalculateAuc(List<RocPoint> scores)
        {
            var positives = scores.Count(point => point.ActualPositive);
            var negatives = scores.Count - positives;
            if (positives == 0 || negatives == 0)
                return double.NaN;

            var ordered = scores.OrderBy(point => point.Score).ToArray();
            var rank = 1.0;
            var positiveRankSum = 0.0;
            for (var i = 0; i < ordered.Length;)
            {
                var end = i + 1;
                while (end < ordered.Length && ordered[end].Score == ordered[i].Score) end++;
                var averageRank = (rank + rank + end - i - 1.0) / 2.0;
                for (var j = i; j < end; j++)
                    if (ordered[j].ActualPositive) positiveRankSum += averageRank;
                rank += end - i;
                i = end;
            }
            return (positiveRankSum - positives * (positives + 1) / 2.0) / (positives * (double)negatives);
        }

        private sealed class RocPoint
        {
            public RocPoint(bool actualPositive, float score)
            {
                ActualPositive = actualPositive;
                Score = score;
            }

            public bool ActualPositive { get; private set; }
            public float Score { get; private set; }
        }

        private static double Divide(int numerator, int denominator)
        {
            return denominator == 0 ? 0.0 : (double)numerator / denominator;
        }

        private static double Divide(double numerator, double denominator)
        {
            return denominator == 0.0 ? 0.0 : numerator / denominator;
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
            if (string.Equals(args[2], "openvino-cpu", StringComparison.OrdinalIgnoreCase))
            {
                executionProvider = ExecutionProvider.OpenVinoCpu;
                return true;
            }
            if (string.Equals(args[2], "openvino-gpu", StringComparison.OrdinalIgnoreCase))
            {
                executionProvider = ExecutionProvider.OpenVinoGpu;
                return true;
            }

            return false;
        }

        private static bool TryParseExecutionProviderName(string value, out ExecutionProvider executionProvider)
        {
            executionProvider = ExecutionProvider.Cpu;
            if (string.Equals(value, "cpu", StringComparison.OrdinalIgnoreCase))
                return true;
            if (string.Equals(value, "directml", StringComparison.OrdinalIgnoreCase))
            {
                executionProvider = ExecutionProvider.DirectML;
                return true;
            }
            if (string.Equals(value, "openvino-cpu", StringComparison.OrdinalIgnoreCase))
            {
                executionProvider = ExecutionProvider.OpenVinoCpu;
                return true;
            }
            if (string.Equals(value, "openvino-gpu", StringComparison.OrdinalIgnoreCase))
            {
                executionProvider = ExecutionProvider.OpenVinoGpu;
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
