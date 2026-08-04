using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Euresys.Open_eVision_22_12;
using OnnxVision.Classification;
using OnnxVision.Detection;
using OnnxVision.Euresys;
using OnnxVision.Imaging;
using OnnxVision.Runtime;

namespace OnnxVision
{
    internal static class Program
    {
        private const int DefaultWarmups = 10;
        private const float DetectionNmsIouThreshold = 0.7f;

        private static readonly HashSet<string> Extensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"
        };

        private static int Main(string[] args)
        {
            bool json = args.Any(item => string.Equals(item, "--json", StringComparison.OrdinalIgnoreCase));
            string[] commandArgs = args
                .Where(item => !string.Equals(item, "--json", StringComparison.OrdinalIgnoreCase))
                .ToArray();

            try
            {
                if (commandArgs.Length == 0 || IsHelp(commandArgs))
                {
                    PrintUsage();
                    return commandArgs.Length == 0 ? 2 : 0;
                }

                int argumentOffset = 0;
                string command = null;
                if (string.Equals(commandArgs[0], "detect", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(commandArgs[0], "benchmark-detect", StringComparison.OrdinalIgnoreCase))
                {
                    command = commandArgs[0];
                    argumentOffset = 1;
                }

                if (commandArgs.Length < argumentOffset + 2)
                    return UsageError(json, "A model path and image source are required.");

                string modelPath = Path.GetFullPath(commandArgs[argumentOffset]);
                if (!File.Exists(modelPath))
                    return UsageError(json, "Model does not exist.");

                var endToEnd = Stopwatch.StartNew();
                var taskDetection = Stopwatch.StartNew();
                OnnxVisionTask task = OnnxModelTaskDetector.Detect(modelPath);
                taskDetection.Stop();

                if (command != null && task != OnnxVisionTask.ObjectDetection)
                    return UsageError(json, "The model does not match the object-detection metadata contract.");

                int defaultRepeats = string.Equals(command, "benchmark-detect",
                    StringComparison.OrdinalIgnoreCase) ? 3 : 1;
                if (task == OnnxVisionTask.ObjectDetection)
                {
                    return RunDetection(commandArgs, json, argumentOffset, defaultRepeats,
                        taskDetection.Elapsed.TotalMilliseconds, endToEnd);
                }

                return RunClassification(commandArgs, json, argumentOffset, defaultRepeats,
                    taskDetection.Elapsed.TotalMilliseconds, endToEnd);
            }
            catch (Exception error)
            {
                PrintFailure(json, error.Message);
                return 1;
            }
        }

        private static int RunClassification(string[] args, bool json, int offset,
            int defaultRepeats, double taskDetectionMilliseconds, Stopwatch endToEnd)
        {
            string modelPath = Path.GetFullPath(args[offset]);
            string testDirectory = Path.GetFullPath(args[offset + 1]);
            if (!Directory.Exists(testDirectory))
                return UsageError(json, "Test directory does not exist.");

            RoiPlacement roi;
            OnnxExecutionProvider[] providers;
            int repeats;
            if (!TryParseClassificationArguments(args, offset, defaultRepeats,
                out providers, out repeats, out roi))
            {
                return UsageError(json,
                    "Usage: OnnxVisionCLI.exe <model.onnx> <test-directory> " +
                    "[provider] [repeats] [roi-x roi-y roi-width roi-height]");
            }

            string[] imagePaths = EnumerateImages(testDirectory);
            if (imagePaths.Length == 0)
                return UsageError(json, "No supported images were found.");

            var construction = Stopwatch.StartNew();
            using (var classifier = new OnnxClassificationModel(modelPath, providers))
            {
                construction.Stop();

                var loadTimer = Stopwatch.StartNew();
                List<LoadedImage> images = LoadImages(imagePaths, classifier.RequiresColorInput);
                loadTimer.Stop();

                try
                {
                    int warmups = Math.Min(DefaultWarmups, images.Count);
                    long warmupModelCallTicks = 0;
                    for (int index = 0; index < warmups; index++)
                    {
                        long callStarted = Stopwatch.GetTimestamp();
                        images[index].Classify(classifier, roi);
                        warmupModelCallTicks += Stopwatch.GetTimestamp() - callStarted;
                    }

                    int executions = checked(images.Count * repeats);
                    var predictions = new List<Dictionary<string, object>>();
                    var rocScores = new List<RocPoint>();
                    var errors = new List<string>();
                    var measuredWall = Stopwatch.StartNew();
                    long modelCallTicks = 0;
                    double onnxInferenceMilliseconds = 0;
                    int correct = 0;
                    int flippedCorrect = 0;
                    int flippedTotal = 0;
                    int normalCorrect = 0;
                    int normalTotal = 0;
                    int truePositives = 0;
                    int falsePositives = 0;
                    int falseNegatives = 0;
                    int trueNegatives = 0;
                    int flippedIndex = FindClassIndex(classifier.ClassNames, "flipped");

                    for (int repeat = 0; repeat < repeats; repeat++)
                    {
                        foreach (LoadedImage image in images)
                        {
                            string expected = new DirectoryInfo(Path.GetDirectoryName(image.Path)).Name;
                            long callStarted = Stopwatch.GetTimestamp();
                            OnnxClassification prediction = image.Classify(classifier, roi);
                            modelCallTicks += Stopwatch.GetTimestamp() - callStarted;
                            onnxInferenceMilliseconds += prediction.InferenceMilliseconds;

                            if (repeat != 0)
                                continue;

                            bool isCorrect = string.Equals(expected, prediction.ClassName,
                                StringComparison.OrdinalIgnoreCase);
                            if (isCorrect)
                                correct++;
                            else
                                errors.Add(string.Format(CultureInfo.InvariantCulture,
                                    "{0} -> {1} ({2:P2})",
                                    MakeRelative(testDirectory, image.Path), prediction.ClassName,
                                    prediction.Confidence));

                            if (string.Equals(expected, "flipped", StringComparison.OrdinalIgnoreCase))
                            {
                                flippedTotal++;
                                if (isCorrect)
                                    flippedCorrect++;
                            }
                            else if (string.Equals(expected, "normal", StringComparison.OrdinalIgnoreCase))
                            {
                                normalTotal++;
                                if (isCorrect)
                                    normalCorrect++;
                            }

                            bool actualPositive = string.Equals(expected, "flipped", StringComparison.OrdinalIgnoreCase);
                            bool predictedPositive = string.Equals(prediction.ClassName, "flipped", StringComparison.OrdinalIgnoreCase);
                            if (actualPositive && predictedPositive)
                                truePositives++;
                            else if (!actualPositive && predictedPositive)
                                falsePositives++;
                            else if (actualPositive)
                                falseNegatives++;
                            else
                                trueNegatives++;

                            if (flippedIndex >= 0 && prediction.Probabilities != null &&
                                prediction.Probabilities.Count > flippedIndex)
                            {
                                rocScores.Add(new RocPoint(actualPositive, prediction.Probabilities[flippedIndex]));
                            }

                            predictions.Add(new Dictionary<string, object>
                            {
                                { "path", MakeRelative(testDirectory, image.Path) },
                                { "expected", expected },
                                { "class_name", prediction.ClassName },
                                { "class_index", prediction.ClassIndex },
                                { "confidence", prediction.Confidence },
                                { "probabilities", prediction.Probabilities }
                            });
                        }
                    }
                    measuredWall.Stop();
                    endToEnd.Stop();

                    if (json)
                    {
                        PrintJson(BuildClassificationReport(modelPath, classifier, providers,
                            imagePaths.Length, warmups, repeats, executions,
                            taskDetectionMilliseconds, construction.Elapsed.TotalMilliseconds,
                            loadTimer.Elapsed.TotalMilliseconds, warmupModelCallTicks,
                            measuredWall.Elapsed.TotalMilliseconds, modelCallTicks,
                            onnxInferenceMilliseconds, endToEnd.Elapsed.TotalMilliseconds,
                            correct, flippedCorrect, flippedTotal, normalCorrect, normalTotal,
                            truePositives, falsePositives, falseNegatives, trueNegatives,
                            rocScores, predictions, errors));
                    }
                    else
                    {
                        PrintClassificationInformation(classifier, imagePaths.Length, roi, providers);
                        PrintClassificationMetrics(correct, imagePaths.Length, flippedCorrect, flippedTotal,
                            normalCorrect, normalTotal, truePositives, falsePositives, falseNegatives,
                            trueNegatives, rocScores);
                        PrintTimingInformation(taskDetectionMilliseconds,
                            imagePaths.Length, repeats, warmups,
                            construction.Elapsed.TotalMilliseconds, loadTimer.Elapsed.TotalMilliseconds,
                            warmupModelCallTicks, modelCallTicks, measuredWall.Elapsed.TotalMilliseconds,
                            endToEnd.Elapsed.TotalMilliseconds, executions);
                        Console.WriteLine("ONNX inference: {0:F3} ms/image",
                            onnxInferenceMilliseconds / executions);
                        foreach (string error in errors)
                            Console.WriteLine("Mismatch: " + error);
                    }
                }
                finally
                {
                    DisposeImages(images);
                }
            }

            return 0;
        }

        private static int RunDetection(string[] args, bool json, int offset,
            int defaultRepeats, double taskDetectionMilliseconds, Stopwatch endToEnd)
        {
            string modelPath = Path.GetFullPath(args[offset]);
            string imageSource = Path.GetFullPath(args[offset + 1]);
            float threshold;
            OnnxExecutionProvider[] providers;
            int repeats;
            if (!TryParseDetectionArguments(args, offset, defaultRepeats,
                out threshold, out repeats, out providers))
            {
                return UsageError(json,
                    "Usage: OnnxVisionCLI.exe <model.onnx> <image-or-directory> " +
                    "[confidence] [repeats] [provider]");
            }

            if (!File.Exists(imageSource) && !Directory.Exists(imageSource))
                return UsageError(json, "Image or image directory does not exist.");

            string[] imagePaths = File.Exists(imageSource)
                ? new[] { imageSource }
                : EnumerateImages(imageSource);
            if (imagePaths.Length == 0)
                return UsageError(json, "No supported images were found.");

            var construction = Stopwatch.StartNew();
            using (var detector = new OnnxObjectDetectionModel(modelPath, null, providers))
            {
                construction.Stop();
                var loadTimer = Stopwatch.StartNew();
                List<LoadedImage> images = LoadImages(imagePaths, detector.RequiresColorInput);
                loadTimer.Stop();

                try
                {
                    int warmups = Math.Min(DefaultWarmups, images.Count);
                    long warmupModelCallTicks = 0;
                    for (int index = 0; index < warmups; index++)
                    {
                        long callStarted = Stopwatch.GetTimestamp();
                        images[index].Detect(detector, threshold, DetectionNmsIouThreshold);
                        warmupModelCallTicks += Stopwatch.GetTimestamp() - callStarted;
                    }

                    int executions = checked(images.Count * repeats);
                    var results = new List<Dictionary<string, object>>();
                    long modelCallTicks = 0;
                    int detectionCount = 0;
                    double confidenceSum = 0;
                    var classCounts = detector.ClassNames.ToDictionary(name => name, name => 0);
                    var measuredWall = Stopwatch.StartNew();
                    for (int repeat = 0; repeat < repeats; repeat++)
                    {
                        foreach (LoadedImage image in images)
                        {
                            long callStarted = Stopwatch.GetTimestamp();
                            IReadOnlyList<OnnxDetection> detections = image.Detect(
                                detector, threshold, DetectionNmsIouThreshold);
                            modelCallTicks += Stopwatch.GetTimestamp() - callStarted;
                            detectionCount += detections.Count;
                            foreach (OnnxDetection detection in detections)
                            {
                                classCounts[detection.ClassName]++;
                                confidenceSum += detection.Confidence;
                            }

                            if (repeat == 0)
                            {
                                results.Add(BuildDetectionImageResult(image.Path, detections));
                                if (!json)
                                {
                                    Console.WriteLine("{0}: {1} detection(s)", image.Path, detections.Count);
                                    foreach (OnnxDetection detection in detections)
                                    {
                                        Console.WriteLine("  {0} {1:F4} [{2:F1}, {3:F1}, {4:F1}, {5:F1}]",
                                            detection.ClassName, detection.Confidence,
                                            detection.X1, detection.Y1, detection.X2, detection.Y2);
                                    }
                                }
                            }
                        }
                    }
                    measuredWall.Stop();
                    endToEnd.Stop();

                    if (json)
                    {
                        var report = BuildBaseReport("detect", modelPath, detector, providers);
                        report["confidence_threshold"] = threshold;
                        report["nms_iou_threshold"] = DetectionNmsIouThreshold;
                        report["images"] = results;
                        report["repeats"] = repeats;
                        report["executions"] = executions;
                        report["warmups"] = warmups;
                        report["detections"] = detectionCount;
                        report["confidence_sum"] = confidenceSum;
                        report["class_counts"] = classCounts;
                        report["timing"] = BuildTimingReport(taskDetectionMilliseconds,
                            construction.Elapsed.TotalMilliseconds, loadTimer.Elapsed.TotalMilliseconds,
                            warmupModelCallTicks, modelCallTicks, measuredWall.Elapsed.TotalMilliseconds,
                            endToEnd.Elapsed.TotalMilliseconds, executions);
                        PrintJson(report);
                    }
                    else
                    {
                        PrintDetectionInformation(detector, providers);
                        PrintTimingInformation(taskDetectionMilliseconds,
                            images.Count, repeats, warmups,
                            construction.Elapsed.TotalMilliseconds, loadTimer.Elapsed.TotalMilliseconds,
                            warmupModelCallTicks, modelCallTicks, measuredWall.Elapsed.TotalMilliseconds,
                            endToEnd.Elapsed.TotalMilliseconds, executions);
                        Console.WriteLine("NMS required: {0}", detector.NmsRequired);
                        Console.WriteLine("Detections: {0}; confidence sum: {1:F6}", detectionCount, confidenceSum);
                        Console.WriteLine("Class counts: " + string.Join(", ",
                            classCounts.Select(item => item.Key + "=" + item.Value)));
                    }
                }
                finally
                {
                    DisposeImages(images);
                }
            }

            return 0;
        }

        private static Dictionary<string, object> BuildClassificationReport(
            string modelPath, OnnxClassificationModel classifier,
            OnnxExecutionProvider[] providers, int imageCount, int warmups, int repeats,
            int executions, double taskDetectionMilliseconds,
            double constructionMilliseconds, double loadMilliseconds,
            long warmupModelCallTicks, double measuredWallMilliseconds,
            long modelCallTicks, double onnxInferenceMilliseconds,
            double endToEndMilliseconds, int correct, int flippedCorrect,
            int flippedTotal, int normalCorrect, int normalTotal, int truePositives,
            int falsePositives, int falseNegatives, int trueNegatives,
            List<RocPoint> rocScores, List<Dictionary<string, object>> predictions,
            List<string> errors)
        {
            var report = BuildBaseReport("classify", modelPath, classifier, providers);
            report["images"] = predictions;
            report["warmups"] = warmups;
            report["repeats"] = repeats;
            report["executions"] = executions;
            report["summary"] = new Dictionary<string, object>
            {
                { "correct", correct },
                { "total", imageCount },
                { "accuracy", Divide(correct, imageCount) },
                { "flipped_correct", flippedCorrect },
                { "flipped_total", flippedTotal },
                { "flipped_recall", Divide(flippedCorrect, flippedTotal) },
                { "normal_correct", normalCorrect },
                { "normal_total", normalTotal },
                { "normal_recall", Divide(normalCorrect, normalTotal) },
                { "true_positives", truePositives },
                { "false_positives", falsePositives },
                { "false_negatives", falseNegatives },
                { "true_negatives", trueNegatives },
                { "roc_auc_flipped_positive", CalculateAuc(rocScores) },
                { "errors", errors }
            };
            report["timing"] = BuildTimingReport(taskDetectionMilliseconds,
                constructionMilliseconds, loadMilliseconds, warmupModelCallTicks,
                modelCallTicks, measuredWallMilliseconds, endToEndMilliseconds, executions);
            ((Dictionary<string, object>)report["timing"])["onnx_inference_milliseconds"] = onnxInferenceMilliseconds;
            ((Dictionary<string, object>)report["timing"])["onnx_inference_milliseconds_per_image"] =
                onnxInferenceMilliseconds / executions;
            return report;
        }

        private static Dictionary<string, object> BuildBaseReport(
            string command, string modelPath, OnnxClassificationModel model,
            OnnxExecutionProvider[] providers)
        {
            var report = new Dictionary<string, object>();
            report["command"] = command;
            report["task"] = "classification";
            report["model"] = modelPath;
            report["contract"] = OnnxVisionContract.ClassificationName;
            report["contract_version"] = OnnxVisionContract.Version;
            report["requested_providers"] = providers.Select(item => item.ToString()).ToArray();
            report["actual_provider"] = model.ActualProvider.ToString();
            report["input"] = new Dictionary<string, object>
            {
                { "pixel_format", model.RequiredPixelFormat.ToString() },
                { "width", model.InputWidth },
                { "height", model.InputHeight },
                { "color", model.RequiresColorInput }
            };
            return report;
        }

        private static Dictionary<string, object> BuildBaseReport(
            string command, string modelPath, OnnxObjectDetectionModel model,
            OnnxExecutionProvider[] providers)
        {
            var report = new Dictionary<string, object>();
            report["command"] = command;
            report["task"] = "object_detection";
            report["model"] = modelPath;
            report["contract"] = OnnxVisionContract.ObjectDetectionName;
            report["contract_version"] = OnnxVisionContract.Version;
            report["requested_providers"] = providers.Select(item => item.ToString()).ToArray();
            report["actual_provider"] = model.ActualProvider.ToString();
            report["nms_required"] = model.NmsRequired;
            report["input"] = new Dictionary<string, object>
            {
                { "description", model.InputDescription },
                { "pixel_format", model.RequiredPixelFormat.ToString() },
                { "width", model.InputWidth },
                { "height", model.InputHeight },
                { "color", model.RequiresColorInput }
            };
            return report;
        }

        private static Dictionary<string, object> BuildTimingReport(
            double taskDetectionMilliseconds, double constructionMilliseconds,
            double loadMilliseconds, long warmupModelCallTicks, long modelCallTicks,
            double measuredWallMilliseconds, double endToEndMilliseconds, int executions)
        {
            double warmupModelCallMilliseconds = TicksToMilliseconds(warmupModelCallTicks);
            double modelCallMilliseconds = TicksToMilliseconds(modelCallTicks);
            return new Dictionary<string, object>
            {
                { "task_detection_milliseconds", taskDetectionMilliseconds },
                { "session_construction_milliseconds", constructionMilliseconds },
                { "image_load_milliseconds", loadMilliseconds },
                { "warmup_model_call_milliseconds", warmupModelCallMilliseconds },
                { "model_call_milliseconds", modelCallMilliseconds },
                { "model_call_milliseconds_per_image", modelCallMilliseconds / executions },
                { "model_call_images_per_second", executions / Math.Max(0.000001, modelCallMilliseconds / 1000.0) },
                { "measured_wall_milliseconds", measuredWallMilliseconds },
                { "measured_wall_milliseconds_per_image", measuredWallMilliseconds / executions },
                { "measured_images_per_second", executions / Math.Max(0.000001, measuredWallMilliseconds / 1000.0) },
                { "end_to_end_milliseconds", endToEndMilliseconds },
                { "end_to_end_milliseconds_per_image", endToEndMilliseconds / executions },
                { "end_to_end_images_per_second", executions / Math.Max(0.000001, endToEndMilliseconds / 1000.0) }
            };
        }

        private static Dictionary<string, object> BuildDetectionImageResult(
            string imagePath, IReadOnlyList<OnnxDetection> detections)
        {
            return new Dictionary<string, object>
            {
                { "path", imagePath },
                { "detections", detections.Select(BuildDetectionResult).ToArray() }
            };
        }

        private static Dictionary<string, object> BuildDetectionResult(OnnxDetection detection)
        {
            return new Dictionary<string, object>
            {
                { "class_name", detection.ClassName },
                { "class_index", detection.ClassIndex },
                { "confidence", detection.Confidence },
                { "x1", detection.X1 },
                { "y1", detection.Y1 },
                { "x2", detection.X2 },
                { "y2", detection.Y2 }
            };
        }

        private static void PrintClassificationInformation(
            OnnxClassificationModel classifier, int imageCount,
            RoiPlacement roi, OnnxExecutionProvider[] providers)
        {
            Console.WriteLine("Provider: {0}", classifier.ActualProvider);
            Console.WriteLine("Requested providers: {0}", FormatProviders(providers));
            Console.WriteLine("Model input: {0}x{1} {2}; test images: {3}",
                classifier.InputWidth, classifier.InputHeight,
                classifier.RequiredPixelFormat, imageCount);
            Console.WriteLine("Class names: " + string.Join(", ", classifier.ClassNames));
            Console.WriteLine("Input region: " + (roi == null ? "full image" : roi.ToString()));
        }

        private static void PrintDetectionInformation(
            OnnxObjectDetectionModel detector, OnnxExecutionProvider[] providers)
        {
            Console.WriteLine("Detection contract: {0} {1}",
                OnnxVisionContract.ObjectDetectionName, OnnxVisionContract.Version);
            Console.WriteLine("Provider: {0}", detector.ActualProvider);
            Console.WriteLine("Requested providers: {0}", FormatProviders(providers));
            Console.WriteLine("Input contract: " + detector.InputDescription);
            Console.WriteLine("NMS required: {0}", detector.NmsRequired);
        }

        private static void PrintTimingInformation(
            double taskDetectionMilliseconds, int imageCount, int repeats, int warmups,
            double constructionMilliseconds, double loadMilliseconds,
            long warmupModelCallTicks, long modelCallTicks,
            double measuredWallMilliseconds, double endToEndMilliseconds, int executions)
        {
            double warmupModelCallMilliseconds = TicksToMilliseconds(warmupModelCallTicks);
            double modelCallMilliseconds = TicksToMilliseconds(modelCallTicks);
            Console.WriteLine("Images: {0}; repeats: {1}; executions: {2}; warmups: {3}",
                imageCount, repeats, executions, warmups);
            Console.WriteLine("Task detection: {0:F3} ms", taskDetectionMilliseconds);
            Console.WriteLine("Session construction: {0:F3} ms", constructionMilliseconds);
            Console.WriteLine("Image load: {0:F3} ms", loadMilliseconds);
            Console.WriteLine("Warmup model call: {0:F3} ms", warmupModelCallMilliseconds);
            Console.WriteLine("Measured wall: {0:F3} ms/image ({1:F2} images/s)",
                measuredWallMilliseconds / executions,
                executions / Math.Max(0.000001, measuredWallMilliseconds / 1000.0));
            Console.WriteLine("Shared model call: {0:F3} ms/image ({1:F2} images/s)",
                modelCallMilliseconds / executions,
                executions / Math.Max(0.000001, modelCallMilliseconds / 1000.0));
            Console.WriteLine("End-to-end: {0:F3} ms/image ({1:F2} images/s)",
                endToEndMilliseconds / executions,
                executions / Math.Max(0.000001, endToEndMilliseconds / 1000.0));
        }

        private static void PrintClassificationMetrics(
            int correct, int total, int flippedCorrect, int flippedTotal,
            int normalCorrect, int normalTotal, int truePositives,
            int falsePositives, int falseNegatives, int trueNegatives,
            List<RocPoint> rocScores)
        {
            Console.WriteLine("Accuracy: {0}/{1} ({2:P2})", correct, total, Divide(correct, total));
            Console.WriteLine("Flipped recall: {0}/{1} ({2:P2})",
                flippedCorrect, flippedTotal, Divide(flippedCorrect, flippedTotal));
            Console.WriteLine("Normal recall: {0}/{1} ({2:P2})",
                normalCorrect, normalTotal, Divide(normalCorrect, normalTotal));
            PrintClassificationMetrics(truePositives, falsePositives, falseNegatives,
                trueNegatives, rocScores);
        }

        private static void PrintClassificationMetrics(
            int truePositives, int falsePositives, int falseNegatives,
            int trueNegatives, List<RocPoint> rocScores)
        {
            double precision = Divide(truePositives, truePositives + falsePositives);
            double recall = Divide(truePositives, truePositives + falseNegatives);
            double f1 = Divide(2.0 * precision * recall, precision + recall);
            double normalPrecision = Divide(trueNegatives, trueNegatives + falseNegatives);
            double normalRecall = Divide(trueNegatives, trueNegatives + falsePositives);
            double normalF1 = Divide(2.0 * normalPrecision * normalRecall,
                normalPrecision + normalRecall);
            Console.WriteLine("Confusion matrix (actual rows / predicted columns):");
            Console.WriteLine("                 flipped  normal");
            Console.WriteLine("actual flipped    {0,7} {1,7}", truePositives, falseNegatives);
            Console.WriteLine("actual normal     {0,7} {1,7}", falsePositives, trueNegatives);
            Console.WriteLine("Flipped precision: {0:P2}; recall: {1:P2}; F1: {2:P2}",
                precision, recall, f1);
            Console.WriteLine("Normal precision: {0:P2}; recall: {1:P2}; F1: {2:P2}",
                normalPrecision, normalRecall, normalF1);
            Console.WriteLine("Macro precision: {0:P2}; macro recall: {1:P2}; macro F1: {2:P2}",
                (precision + normalPrecision) / 2.0,
                (recall + normalRecall) / 2.0,
                (f1 + normalF1) / 2.0);
            Console.WriteLine("ROC AUC (flipped positive): {0:F4}", CalculateAuc(rocScores));
        }

        private static bool TryParseClassificationArguments(string[] args, int offset,
            int defaultRepeats, out OnnxExecutionProvider[] providers, out int repeats,
            out RoiPlacement roi)
        {
            providers = new[] { OnnxExecutionProvider.Cpu };
            repeats = defaultRepeats;
            roi = null;

            int index = offset + 2;
            int repeatArguments = 0;
            int providerArguments = 0;
            while (index < args.Length)
            {
                if (args.Length - index == 4 && TryParseRoi(args, index, out roi))
                {
                    index += 4;
                    continue;
                }

                OnnxExecutionProvider[] parsedProviders;
                if (TryParseProvider(args[index], out parsedProviders))
                {
                    if (++providerArguments > 1)
                        return false;
                    providers = parsedProviders;
                    index++;
                    continue;
                }

                int parsedRepeats;
                if (!TryParsePositiveInteger(args[index], out parsedRepeats) || ++repeatArguments > 1)
                    return false;
                repeats = parsedRepeats;
                index++;
            }

            return repeatArguments <= 1 && providerArguments <= 1;
        }

        private static bool TryParseDetectionArguments(string[] args, int offset,
            int defaultRepeats, out float threshold, out int repeats,
            out OnnxExecutionProvider[] providers)
        {
            threshold = 0.5f;
            repeats = defaultRepeats;
            providers = new[] { OnnxExecutionProvider.Cpu };
            bool thresholdSpecified = false;
            bool repeatsSpecified = false;
            bool providerSpecified = false;

            int index = offset + 2;
            if (args.Length - index > 3)
                return false;

            while (index < args.Length)
            {
                OnnxExecutionProvider[] parsedProviders;
                if (TryParseProvider(args[index], out parsedProviders))
                {
                    if (providerSpecified)
                        return false;
                    providers = parsedProviders;
                    providerSpecified = true;
                    index++;
                    continue;
                }

                float parsedThreshold;
                if (!thresholdSpecified && TryParseThreshold(args[index], out parsedThreshold))
                {
                    threshold = parsedThreshold;
                    thresholdSpecified = true;
                    index++;
                    continue;
                }

                int parsedRepeats;
                if (!repeatsSpecified && TryParsePositiveInteger(args[index], out parsedRepeats))
                {
                    repeats = parsedRepeats;
                    repeatsSpecified = true;
                    index++;
                    continue;
                }

                return false;
            }

            return true;
        }

        private static bool TryParseProvider(
            string value, out OnnxExecutionProvider[] providers)
        {
            OnnxExecutionProvider provider;
            if (string.IsNullOrWhiteSpace(value) ||
                string.Equals(value, "cpu", StringComparison.OrdinalIgnoreCase))
            {
                providers = new[] { OnnxExecutionProvider.Cpu };
                return true;
            }

            if (string.Equals(value, "openvino-cpu", StringComparison.OrdinalIgnoreCase))
                provider = OnnxExecutionProvider.OpenVinoCpu;
            else if (string.Equals(value, "openvino-gpu", StringComparison.OrdinalIgnoreCase))
                provider = OnnxExecutionProvider.OpenVinoGpu;
            else
            {
                providers = null;
                return false;
            }

            providers = new[] { provider, OnnxExecutionProvider.Cpu };
            return true;
        }

        private static bool TryParseThreshold(string value, out float threshold)
        {
            threshold = 0.5f;
            if (string.IsNullOrWhiteSpace(value))
                return true;
            return float.TryParse(value, NumberStyles.Float,
                CultureInfo.InvariantCulture, out threshold) &&
                threshold >= 0 && threshold <= 1;
        }

        private static bool TryParsePositiveInteger(string value, out int result)
        {
            return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out result) &&
                result > 0;
        }

        private static bool TryParseRoi(string[] args, int offset, out RoiPlacement placement)
        {
            placement = null;
            int x;
            int y;
            int width;
            int height;
            if (!int.TryParse(args[offset], out x) ||
                !int.TryParse(args[offset + 1], out y) ||
                !int.TryParse(args[offset + 2], out width) ||
                !int.TryParse(args[offset + 3], out height) ||
                width <= 0 || height <= 0)
            {
                return false;
            }

            placement = new RoiPlacement(x, y, width, height);
            return true;
        }

        private static string[] EnumerateImages(string directory)
        {
            return Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories)
                .Where(path => Extensions.Contains(Path.GetExtension(path)))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }

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

        private static Dictionary<string, object> BuildFailure(string message)
        {
            return new Dictionary<string, object>
            {
                { "error", message }
            };
        }

        private static int UsageError(bool json, string message)
        {
            if (json)
                PrintJson(BuildFailure(message));
            else
            {
                Console.Error.WriteLine(message);
                PrintUsage();
            }
            return 2;
        }

        private static void PrintFailure(bool json, string message)
        {
            if (json)
                PrintJson(BuildFailure(message));
            else
                Console.Error.WriteLine("Error: " + message);
        }

        private static void PrintUsage()
        {
            Console.WriteLine("Usage:");
            Console.WriteLine("  OnnxVisionCLI.exe <model.onnx> <image-or-directory> [task options] [--json]");
            Console.WriteLine("  Classification options: [provider] [repeats] [roi-x roi-y roi-width roi-height]");
            Console.WriteLine("  Detection options: [confidence] [repeats] [provider]");
            Console.WriteLine("  'detect' remains an optional object-detection command alias.");
            Console.WriteLine("Providers: cpu, openvino-cpu, openvino-gpu");
            Console.WriteLine("Models are classified automatically from their ONNX metadata contract.");
        }

        private static bool IsHelp(string[] args)
        {
            return args.Length == 1 &&
                (string.Equals(args[0], "--help", StringComparison.OrdinalIgnoreCase) ||
                 string.Equals(args[0], "-h", StringComparison.OrdinalIgnoreCase) ||
                 string.Equals(args[0], "help", StringComparison.OrdinalIgnoreCase));
        }

        private static string FormatProviders(IEnumerable<OnnxExecutionProvider> providers)
        {
            return string.Join(", ", providers.Select(item => item.ToString()).ToArray());
        }

        private static string MakeRelative(string root, string path)
        {
            Uri rootUri = new Uri(root.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar);
            return Uri.UnescapeDataString(rootUri.MakeRelativeUri(new Uri(path)).ToString())
                .Replace('/', Path.DirectorySeparatorChar);
        }

        private static int FindClassIndex(IReadOnlyList<string> classNames, string name)
        {
            for (int index = 0; index < classNames.Count; index++)
            {
                if (string.Equals(classNames[index], name, StringComparison.OrdinalIgnoreCase))
                    return index;
            }
            return -1;
        }

        private static double TicksToMilliseconds(long ticks)
        {
            return ticks * 1000.0 / Stopwatch.Frequency;
        }

        private static double Divide(int numerator, int denominator)
        {
            return denominator == 0 ? 0.0 : (double)numerator / denominator;
        }

        private static double Divide(double numerator, double denominator)
        {
            return denominator == 0.0 ? 0.0 : numerator / denominator;
        }

        private static double CalculateAuc(List<RocPoint> scores)
        {
            int positives = scores.Count(point => point.ActualPositive);
            int negatives = scores.Count - positives;
            if (positives == 0 || negatives == 0)
                return double.NaN;

            RocPoint[] ordered = scores.OrderBy(point => point.Score).ToArray();
            double rank = 1.0;
            double positiveRankSum = 0.0;
            for (int index = 0; index < ordered.Length;)
            {
                int end = index + 1;
                while (end < ordered.Length && ordered[end].Score == ordered[index].Score)
                    end++;
                double averageRank = (rank + rank + end - index - 1.0) / 2.0;
                for (int item = index; item < end; item++)
                {
                    if (ordered[item].ActualPositive)
                        positiveRankSum += averageRank;
                }
                rank += end - index;
                index = end;
            }
            return (positiveRankSum - positives * (positives + 1) / 2.0) /
                (positives * (double)negatives);
        }

        private static void PrintJson(object value)
        {
            Console.WriteLine(ToJson(value));
        }

        private static string ToJson(object value)
        {
            var builder = new StringBuilder();
            WriteJsonValue(builder, value);
            return builder.ToString();
        }

        private static void WriteJsonValue(StringBuilder builder, object value)
        {
            if (value == null)
            {
                builder.Append("null");
                return;
            }

            string text = value as string;
            if (text != null)
            {
                WriteJsonString(builder, text);
                return;
            }

            if (value is bool)
            {
                builder.Append((bool)value ? "true" : "false");
                return;
            }

            if (value is char)
            {
                WriteJsonString(builder, value.ToString());
                return;
            }

            if (value is byte || value is sbyte || value is short || value is ushort ||
                value is int || value is uint || value is long || value is ulong ||
                value is float || value is double || value is decimal)
            {
                double number = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                if (double.IsNaN(number) || double.IsInfinity(number))
                    builder.Append("null");
                else
                    builder.Append(number.ToString("R", CultureInfo.InvariantCulture));
                return;
            }

            IDictionary dictionary = value as IDictionary;
            if (dictionary != null)
            {
                builder.Append('{');
                bool first = true;
                foreach (DictionaryEntry entry in dictionary)
                {
                    if (!first)
                        builder.Append(',');
                    first = false;
                    WriteJsonString(builder, Convert.ToString(entry.Key, CultureInfo.InvariantCulture));
                    builder.Append(':');
                    WriteJsonValue(builder, entry.Value);
                }
                builder.Append('}');
                return;
            }

            IEnumerable enumerable = value as IEnumerable;
            if (enumerable != null)
            {
                builder.Append('[');
                bool first = true;
                foreach (object item in enumerable)
                {
                    if (!first)
                        builder.Append(',');
                    first = false;
                    WriteJsonValue(builder, item);
                }
                builder.Append(']');
                return;
            }

            throw new InvalidOperationException("Unsupported JSON value type: " + value.GetType().FullName);
        }

        private static void WriteJsonString(StringBuilder builder, string value)
        {
            builder.Append('"');
            foreach (char character in value)
            {
                switch (character)
                {
                    case '"': builder.Append("\\\""); break;
                    case '\\': builder.Append("\\\\"); break;
                    case '\b': builder.Append("\\b"); break;
                    case '\f': builder.Append("\\f"); break;
                    case '\n': builder.Append("\\n"); break;
                    case '\r': builder.Append("\\r"); break;
                    case '\t': builder.Append("\\t"); break;
                    default:
                        if (character < 32)
                            builder.Append("\\u").Append(((int)character).ToString("x4", CultureInfo.InvariantCulture));
                        else
                            builder.Append(character);
                        break;
                }
            }
            builder.Append('"');
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

            public Rectangle ToRectangle()
            {
                return new Rectangle(X, Y, Width, Height);
            }

            public override string ToString()
            {
                return string.Format(CultureInfo.InvariantCulture,
                    "ROI ({0}, {1}, {2}, {3})", X, Y, Width, Height);
            }
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
    }
}
