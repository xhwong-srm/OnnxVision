using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
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
            bool json = args.Any(IsJsonFlag);
            string[] commandArgs = args
                .Where(item => !IsJsonFlag(item))
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
            string imageSource = Path.GetFullPath(args[offset + 1]);

            RoiPlacement roi;
            OnnxExecutionProvider[] providers;
            int repeats;
            InputOptions inputOptions;
            if (!TryParseClassificationArguments(args, offset, defaultRepeats,
                out providers, out repeats, out roi, out inputOptions))
            {
                return UsageError(json,
                    "Usage: OnnxVisionCLI.exe <model.onnx> <image-file|image-directory|dataset> " +
                    "[provider] [repeats] [roi-x roi-y roi-width roi-height] " +
                    "[-dataset] [-validate] [-set train|val|test]");
            }

            ClassificationInput input;
            try
            {
                input = LoadClassificationInput(imageSource, inputOptions);
            }
            catch (Exception error)
            {
                return UsageError(json, error.Message);
            }

            if ((inputOptions.Validate || inputOptions.Set != null) && !input.IsDataset)
                return UsageError(json, "-validate and -set require an ImageNet-style classification dataset.");

            string[] imagePaths = input.Samples.Select(item => item.Path).ToArray();
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
                    ClassificationValidationMetrics validation = inputOptions.Validate
                        ? new ClassificationValidationMetrics(classifier.ClassNames,
                            input.DatasetFormat, input.DatasetSplit)
                        : null;
                    var samplesByPath = input.Samples.ToDictionary(item => item.Path,
                        StringComparer.OrdinalIgnoreCase);
                    int labeledImageCount = input.Samples.Count(item => item.ExpectedClassName != null);
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
                            ClassificationSample sample = samplesByPath[image.Path];
                            string expected = sample.ExpectedClassName;
                            long callStarted = Stopwatch.GetTimestamp();
                            OnnxClassification prediction = image.Classify(classifier, roi);
                            modelCallTicks += Stopwatch.GetTimestamp() - callStarted;
                            onnxInferenceMilliseconds += prediction.InferenceMilliseconds;

                            if (repeat != 0)
                                continue;

                            bool hasExpected = !string.IsNullOrWhiteSpace(expected);
                            bool isCorrect = hasExpected && string.Equals(expected, prediction.ClassName,
                                StringComparison.OrdinalIgnoreCase);
                            if (hasExpected)
                            {
                                if (isCorrect)
                                    correct++;
                                else
                                    errors.Add(string.Format(CultureInfo.InvariantCulture,
                                        "{0} -> {1} ({2:P2})",
                                        image.Path, prediction.ClassName, prediction.Confidence));

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

                                if (validation != null)
                                    validation.Add(expected, prediction);
                            }

                            predictions.Add(new Dictionary<string, object>
                            {
                                { "path", image.Path },
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
                        var report = BuildClassificationReport(modelPath, classifier, providers,
                            imagePaths.Length, labeledImageCount, warmups, repeats, executions,
                            taskDetectionMilliseconds, construction.Elapsed.TotalMilliseconds,
                            loadTimer.Elapsed.TotalMilliseconds, warmupModelCallTicks,
                            measuredWall.Elapsed.TotalMilliseconds, modelCallTicks,
                            onnxInferenceMilliseconds, endToEnd.Elapsed.TotalMilliseconds,
                            correct, flippedCorrect, flippedTotal, normalCorrect, normalTotal,
                            truePositives, falsePositives, falseNegatives, trueNegatives,
                            rocScores, predictions, errors);
                        if (validation != null)
                            report["validation"] = validation.ToReport();
                        PrintJson(report);
                    }
                    else
                    {
                        PrintClassificationInformation(classifier, imagePaths.Length, roi, providers,
                            input.IsDataset, input.DatasetFormat, input.DatasetSplit);
                        if (validation != null)
                            PrintClassificationValidation(validation);
                        else if (labeledImageCount > 0)
                            PrintClassificationMetrics(correct, labeledImageCount, flippedCorrect, flippedTotal,
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
            InputOptions inputOptions;
            if (!TryParseDetectionArguments(args, offset, defaultRepeats,
                out threshold, out repeats, out providers, out inputOptions))
            {
                return UsageError(json,
                    "Usage: OnnxVisionCLI.exe <model.onnx> <image-file|image-directory|COCO-dataset> " +
                    "[confidence] [repeats] [provider] [-dataset] [-validate] " +
                    "[-set train|val|test]");
            }

            DetectionInput input;
            try
            {
                input = LoadDetectionInput(imageSource, inputOptions);
            }
            catch (Exception error)
            {
                return UsageError(json, error.Message);
            }

            if ((inputOptions.Validate || inputOptions.Set != null) && !input.IsDataset)
                return UsageError(json, "-validate and -set require a COCO detection dataset.");

            string[] imagePaths = input.Samples.Select(item => item.Path).ToArray();
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
                    DetectionValidationMetrics validation = inputOptions.Validate
                        ? new DetectionValidationMetrics(detector.ClassNames,
                            input.DatasetFormat, input.DatasetSplit)
                        : null;
                    var samplesByPath = input.Samples.ToDictionary(item => item.Path,
                        StringComparer.OrdinalIgnoreCase);
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
                                if (validation != null)
                                    validation.Add(image.Path, samplesByPath[image.Path].GroundTruths, detections);
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
                        if (validation != null)
                            report["validation"] = validation.ToReport();
                        report["timing"] = BuildTimingReport(taskDetectionMilliseconds,
                            construction.Elapsed.TotalMilliseconds, loadTimer.Elapsed.TotalMilliseconds,
                            warmupModelCallTicks, modelCallTicks, measuredWall.Elapsed.TotalMilliseconds,
                            endToEnd.Elapsed.TotalMilliseconds, executions);
                        PrintJson(report);
                    }
                    else
                    {
                        PrintDetectionInformation(detector, providers, input.IsDataset,
                            input.DatasetFormat, input.DatasetSplit);
                        if (validation != null)
                            PrintDetectionValidation(validation);
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
            OnnxExecutionProvider[] providers, int imageCount, int labeledImageCount,
            int warmups, int repeats,
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
                { "total", labeledImageCount },
                { "accuracy", Divide(correct, labeledImageCount) },
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
            RoiPlacement roi, OnnxExecutionProvider[] providers,
            bool isDataset, string datasetFormat, string datasetSplit)
        {
            Console.WriteLine("Provider: {0}", classifier.ActualProvider);
            Console.WriteLine("Requested providers: {0}", FormatProviders(providers));
            Console.WriteLine("Model input: {0}x{1} {2}; images: {3}",
                classifier.InputWidth, classifier.InputHeight,
                classifier.RequiredPixelFormat, imageCount);
            if (isDataset)
                Console.WriteLine("Dataset: {0}; set: {1}", datasetFormat, datasetSplit);
            Console.WriteLine("Class names: " + string.Join(", ", classifier.ClassNames));
            Console.WriteLine("Input region: " + (roi == null ? "full image" : roi.ToString()));
        }

        private static void PrintDetectionInformation(
            OnnxObjectDetectionModel detector, OnnxExecutionProvider[] providers,
            bool isDataset, string datasetFormat, string datasetSplit)
        {
            Console.WriteLine("Detection contract: {0} {1}",
                OnnxVisionContract.ObjectDetectionName, OnnxVisionContract.Version);
            Console.WriteLine("Provider: {0}", detector.ActualProvider);
            Console.WriteLine("Requested providers: {0}", FormatProviders(providers));
            Console.WriteLine("Input contract: " + detector.InputDescription);
            Console.WriteLine("NMS required: {0}", detector.NmsRequired);
            if (isDataset)
                Console.WriteLine("Dataset: {0}; set: {1}", datasetFormat, datasetSplit);
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

        private static void PrintClassificationValidation(ClassificationValidationMetrics metrics)
        {
            Dictionary<string, object> report = metrics.ToReport();
            Console.WriteLine("Validation ({0}, {1}): {2}/{3} top-1 accuracy ({4:P2})",
                report["format"], report["set"], report["correct"], report["images"],
                Convert.ToDouble(report["top1_accuracy"], CultureInfo.InvariantCulture));
            Console.WriteLine("Macro precision: {0:P2}; macro recall: {1:P2}; macro F1: {2:P2}",
                Convert.ToDouble(report["macro_precision"], CultureInfo.InvariantCulture),
                Convert.ToDouble(report["macro_recall"], CultureInfo.InvariantCulture),
                Convert.ToDouble(report["macro_f1"], CultureInfo.InvariantCulture));
            foreach (Dictionary<string, object> item in (IEnumerable<Dictionary<string, object>>)report["per_class"])
            {
                Console.WriteLine("  {0}: support={1}; precision={2:P2}; recall={3:P2}; F1={4:P2}",
                    item["class_name"], item["support"],
                    Convert.ToDouble(item["precision"], CultureInfo.InvariantCulture),
                    Convert.ToDouble(item["recall"], CultureInfo.InvariantCulture),
                    Convert.ToDouble(item["f1"], CultureInfo.InvariantCulture));
            }
        }

        private static void PrintDetectionValidation(DetectionValidationMetrics metrics)
        {
            Dictionary<string, object> report = metrics.ToReport();
            Console.WriteLine("Validation ({0}, {1}): {2} image(s), {3} ground-truth box(es)",
                report["format"], report["set"], report["images"], report["ground_truth_boxes"]);
            Console.WriteLine("IoU 0.50 precision: {0:P2}; recall: {1:P2}; F1: {2:P2}",
                Convert.ToDouble(report["precision"], CultureInfo.InvariantCulture),
                Convert.ToDouble(report["recall"], CultureInfo.InvariantCulture),
                Convert.ToDouble(report["f1"], CultureInfo.InvariantCulture));
            Console.WriteLine("mAP50: {0:P2}; mAP50-95: {1:P2}",
                Convert.ToDouble(report["map50"], CultureInfo.InvariantCulture),
                Convert.ToDouble(report["map50_95"], CultureInfo.InvariantCulture));
            foreach (Dictionary<string, object> item in (IEnumerable<Dictionary<string, object>>)report["per_class"])
            {
                Console.WriteLine("  {0}: GT={1}; TP={2}; FP={3}; FN={4}; AP50={5:P2}; AP50-95={6:P2}",
                    item["class_name"], item["ground_truth"], item["true_positives"],
                    item["false_positives"], item["false_negatives"],
                    Convert.ToDouble(item["ap50"], CultureInfo.InvariantCulture),
                    Convert.ToDouble(item["ap50_95"], CultureInfo.InvariantCulture));
            }
        }

        private static bool TryParseClassificationArguments(string[] args, int offset,
            int defaultRepeats, out OnnxExecutionProvider[] providers, out int repeats,
            out RoiPlacement roi, out InputOptions inputOptions)
        {
            providers = new[] { OnnxExecutionProvider.Cpu };
            repeats = defaultRepeats;
            roi = null;
            inputOptions = new InputOptions();

            int index = offset + 2;
            int repeatArguments = 0;
            int providerArguments = 0;
            while (index < args.Length)
            {
                if (TryParseInputOption(args, ref index, inputOptions))
                    continue;

                if (IsFlag(args[index], "roi"))
                {
                    if (roi != null || index + 4 >= args.Length ||
                        !TryParseRoi(args, index + 1, out roi))
                    {
                        return false;
                    }
                    index += 5;
                    continue;
                }

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
            out OnnxExecutionProvider[] providers, out InputOptions inputOptions)
        {
            threshold = 0.5f;
            repeats = defaultRepeats;
            providers = new[] { OnnxExecutionProvider.Cpu };
            inputOptions = new InputOptions();
            bool thresholdSpecified = false;
            bool repeatsSpecified = false;
            bool providerSpecified = false;

            int index = offset + 2;
            while (index < args.Length)
            {
                if (TryParseInputOption(args, ref index, inputOptions))
                    continue;

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

        private static bool TryParseInputOption(string[] args, ref int index,
            InputOptions inputOptions)
        {
            string value = args[index];
            if (IsFlag(value, "validate"))
            {
                inputOptions.Validate = true;
                index++;
                return true;
            }

            if (IsFlag(value, "dataset"))
            {
                inputOptions.ForceDataset = true;
                index++;
                return true;
            }

            string set;
            if (TryParseInlineOption(value, "set", out set))
            {
                if (!TrySetDatasetSplit(inputOptions, set))
                    return false;
                index++;
                return true;
            }

            if (IsFlag(value, "set"))
            {
                if (index + 1 >= args.Length ||
                    !TrySetDatasetSplit(inputOptions, args[index + 1]))
                {
                    return false;
                }
                index += 2;
                return true;
            }

            return false;
        }

        private static bool TrySetDatasetSplit(InputOptions inputOptions, string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return false;
            string split = value.Trim().ToLowerInvariant();
            if (split == "valid" || split == "validation")
                split = "val";
            if (split != "train" && split != "val" && split != "test")
                return false;
            if (inputOptions.Set != null)
                return false;
            inputOptions.Set = split;
            return true;
        }

        private static bool TryParseInlineOption(string value, string name, out string optionValue)
        {
            optionValue = null;
            string prefix = "-" + name + "=";
            string longPrefix = "--" + name + "=";
            if (value.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                optionValue = value.Substring(prefix.Length);
            else if (value.StartsWith(longPrefix, StringComparison.OrdinalIgnoreCase))
                optionValue = value.Substring(longPrefix.Length);
            else
                return false;
            return true;
        }

        private static bool IsFlag(string value, string name)
        {
            return string.Equals(value, "-" + name, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(value, "--" + name, StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsJsonFlag(string value)
        {
            return IsFlag(value, "json");
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

        private static ClassificationInput LoadClassificationInput(string source,
            InputOptions options)
        {
            if (File.Exists(source))
            {
                if (options.ForceDataset || options.Validate || options.Set != null)
                    throw new InvalidOperationException(
                        "A classification dataset must be a directory; a single image cannot be validated.");
                if (!Extensions.Contains(Path.GetExtension(source)))
                    throw new InvalidOperationException("The input file is not a supported image.");
                return new ClassificationInput(
                    new List<ClassificationSample> { new ClassificationSample(source, null) },
                    false, null, null);
            }

            if (!Directory.Exists(source))
                throw new DirectoryNotFoundException("Image or classification dataset does not exist: " + source);

            bool isDataset = options.ForceDataset || IsClassificationDatasetRoot(source);
            if (!isDataset)
            {
                return new ClassificationInput(
                    EnumerateImages(source).Select(path => new ClassificationSample(path, null)).ToList(),
                    false, null, null);
            }

            string split;
            string splitDirectory = ResolveClassificationSplitDirectory(source, options.Set, out split);
            List<ClassificationSample> samples = LoadClassificationSamples(splitDirectory);
            return new ClassificationInput(samples, true, "imagenet", split);
        }

        private static bool IsClassificationDatasetRoot(string root)
        {
            if (FindCocoAnnotation(root, null) != null)
                return false;
            return HasClassDirectoriesWithImages(root) ||
                new[] { "train", "val", "valid", "test" }
                    .Select(split => Path.Combine(root, split))
                    .Any(path => Directory.Exists(path) && HasClassDirectoriesWithImages(path));
        }

        private static bool HasClassDirectoriesWithImages(string root)
        {
            if (!Directory.Exists(root))
                return false;
            foreach (string directory in Directory.EnumerateDirectories(root))
            {
                if (Directory.EnumerateFiles(directory, "*", SearchOption.TopDirectoryOnly)
                    .Any(path => Extensions.Contains(Path.GetExtension(path))))
                {
                    return true;
                }
            }
            return false;
        }

        private static string ResolveClassificationSplitDirectory(string root,
            string requestedSplit, out string selectedSplit)
        {
            selectedSplit = requestedSplit;
            if (requestedSplit == null)
            {
                foreach (string candidate in new[] { "val", "valid", "train", "test" })
                {
                    string candidatePath = Path.Combine(root, candidate);
                    if (Directory.Exists(candidatePath) && HasClassDirectoriesWithImages(candidatePath))
                    {
                        selectedSplit = candidate == "valid" ? "val" : candidate;
                        return candidatePath;
                    }
                }

                if (HasClassDirectoriesWithImages(root))
                {
                    selectedSplit = "root";
                    return root;
                }

                throw new InvalidOperationException(
                    "The classification dataset does not contain train, val, or test class folders.");
            }

            string splitDirectory = Path.Combine(root, requestedSplit);
            if (requestedSplit == "val" && !Directory.Exists(splitDirectory))
                splitDirectory = Path.Combine(root, "valid");
            if (!Directory.Exists(splitDirectory) || !HasClassDirectoriesWithImages(splitDirectory))
            {
                throw new InvalidOperationException(string.Format(CultureInfo.InvariantCulture,
                    "The classification dataset does not contain a labeled '{0}' split.",
                    requestedSplit));
            }
            return splitDirectory;
        }

        private static List<ClassificationSample> LoadClassificationSamples(string root)
        {
            var samples = new List<ClassificationSample>();
            foreach (string classDirectory in Directory.EnumerateDirectories(root)
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
            {
                string className = new DirectoryInfo(classDirectory).Name;
                foreach (string imagePath in EnumerateImages(classDirectory))
                    samples.Add(new ClassificationSample(imagePath, className));
            }
            return samples;
        }

        private static DetectionInput LoadDetectionInput(string source, InputOptions options)
        {
            if (File.Exists(source) && !IsJsonFile(source))
            {
                if (options.ForceDataset || options.Validate || options.Set != null)
                    throw new InvalidOperationException(
                        "A COCO dataset is required for -validate and -set; a single image has no labels.");
                if (!Extensions.Contains(Path.GetExtension(source)))
                    throw new InvalidOperationException("The input file is not a supported image.");
                return new DetectionInput(
                    new List<DetectionSample> { new DetectionSample(source, new List<GroundTruthDetection>()) },
                    false, null, null, null);
            }

            bool datasetRequested = options.ForceDataset || IsCocoDatasetSource(source);
            if (datasetRequested)
            {
                string root = Directory.Exists(source) ? source : InferCocoRoot(source);
                string annotationPath = Directory.Exists(source) && IsJsonFile(source)
                    ? source
                    : FindCocoAnnotation(source, options.Set);
                if (annotationPath == null)
                    throw new InvalidOperationException(
                        "The COCO dataset does not contain annotations for the requested split.");

                string split = options.Set ?? InferCocoSplit(annotationPath);
                return LoadCocoInput(root, annotationPath, split);
            }

            if (!Directory.Exists(source))
                throw new DirectoryNotFoundException("Image or COCO dataset does not exist: " + source);
            if (options.Validate || options.Set != null)
                throw new InvalidOperationException("-validate and -set require a COCO detection dataset.");

            return new DetectionInput(
                EnumerateImages(source)
                    .Select(path => new DetectionSample(path, new List<GroundTruthDetection>()))
                    .ToList(), false, null, null, null);
        }

        private static bool IsCocoDatasetSource(string source)
        {
            if (File.Exists(source))
                return IsJsonFile(source);
            return Directory.Exists(source) && FindCocoAnnotation(source, null) != null;
        }

        private static bool IsJsonFile(string path)
        {
            return string.Equals(Path.GetExtension(path), ".json",
                StringComparison.OrdinalIgnoreCase);
        }

        private static string FindCocoAnnotation(string source, string requestedSplit)
        {
            if (File.Exists(source))
                return IsJsonFile(source) ? source : null;
            if (!Directory.Exists(source))
                return null;

            string[] splits = requestedSplit == null
                ? new[] { "val", "train", "test" }
                : new[] { requestedSplit };
            foreach (string split in splits)
            {
                foreach (string candidate in CocoAnnotationCandidates(source, split))
                {
                    if (File.Exists(candidate))
                        return candidate;
                }
            }
            return null;
        }

        private static IEnumerable<string> CocoAnnotationCandidates(string root, string split)
        {
            string canonical = split == "valid" ? "val" : split;
            string[] splitNames = canonical == "val"
                ? new[] { "val", "valid" }
                : new[] { canonical };
            foreach (string name in splitNames)
            {
                yield return Path.Combine(root, "annotations", "instances_" + name + ".json");
                yield return Path.Combine(root, "annotations", "instances_" + name + "2017.json");
                yield return Path.Combine(root, "annotations", "instances_" + name + "_2017.json");
                yield return Path.Combine(root, name, "_annotations.coco.json");
                yield return Path.Combine(root, name, "annotations.json");
                yield return Path.Combine(root, "instances_" + name + ".json");
                yield return Path.Combine(root, name + ".json");
            }

            string splitDirectory = Path.Combine(root, canonical);
            if (canonical == "val" && !Directory.Exists(splitDirectory))
                splitDirectory = Path.Combine(root, "valid");
            if (File.Exists(Path.Combine(splitDirectory, "_annotations.coco.json")))
                yield return Path.Combine(splitDirectory, "_annotations.coco.json");

            string annotationDirectory = Path.Combine(root, "annotations");
            if (Directory.Exists(annotationDirectory))
            {
                foreach (string path in Directory.EnumerateFiles(annotationDirectory, "*.json",
                    SearchOption.TopDirectoryOnly).OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
                {
                    string name = Path.GetFileName(path).ToLowerInvariant();
                    if (name.Contains(canonical) || (canonical == "val" && name.Contains("valid")))
                        yield return path;
                }
            }
        }

        private static string InferCocoRoot(string annotationPath)
        {
            string parent = Path.GetDirectoryName(annotationPath);
            if (parent != null && string.Equals(new DirectoryInfo(parent).Name, "annotations",
                StringComparison.OrdinalIgnoreCase))
            {
                return Directory.GetParent(parent).FullName;
            }
            return parent;
        }

        private static string InferCocoSplit(string annotationPath)
        {
            string name = Path.GetFileName(annotationPath).ToLowerInvariant();
            if (name.Contains("val") || name.Contains("valid"))
                return "val";
            if (name.Contains("test"))
                return "test";
            string parent = Path.GetDirectoryName(annotationPath);
            if (parent != null)
            {
                string parentName = new DirectoryInfo(parent).Name.ToLowerInvariant();
                if (parentName == "val" || parentName == "valid")
                    return "val";
                if (parentName == "test")
                    return "test";
            }
            return "train";
        }

        private static DetectionInput LoadCocoInput(string root, string annotationPath,
            string split)
        {
            CocoDocument document;
            var serializer = new DataContractJsonSerializer(typeof(CocoDocument));
            using (var stream = File.OpenRead(annotationPath))
                document = (CocoDocument)serializer.ReadObject(stream);
            if (document == null || document.Images == null || document.Annotations == null ||
                document.Categories == null)
            {
                throw new InvalidOperationException(
                    "COCO annotations must contain images, annotations, and categories arrays.");
            }

            var categoryNames = new Dictionary<long, string>();
            foreach (CocoCategory category in document.Categories)
            {
                if (category == null || string.IsNullOrWhiteSpace(category.Name))
                    throw new InvalidOperationException("COCO contains an empty category name.");
                if (categoryNames.ContainsKey(category.Id))
                    throw new InvalidOperationException("COCO contains duplicate category IDs.");
                categoryNames.Add(category.Id, category.Name.Trim());
            }

            var annotationsByImage = new Dictionary<long, List<GroundTruthDetection>>();
            foreach (CocoAnnotation annotation in document.Annotations)
            {
                string className;
                if (!categoryNames.TryGetValue(annotation.CategoryId, out className))
                    throw new InvalidOperationException("COCO annotation references an unknown category ID.");
                if (annotation.BoundingBox == null || annotation.BoundingBox.Length < 4)
                    throw new InvalidOperationException("COCO contains an annotation without a valid bbox.");
                double x = annotation.BoundingBox[0];
                double y = annotation.BoundingBox[1];
                double width = annotation.BoundingBox[2];
                double height = annotation.BoundingBox[3];
                if (width <= 0 || height <= 0)
                    continue;
                List<GroundTruthDetection> imageAnnotations;
                if (!annotationsByImage.TryGetValue(annotation.ImageId, out imageAnnotations))
                {
                    imageAnnotations = new List<GroundTruthDetection>();
                    annotationsByImage.Add(annotation.ImageId, imageAnnotations);
                }
                imageAnnotations.Add(new GroundTruthDetection(className,
                    (float)x, (float)y, (float)(x + width), (float)(y + height)));
            }

            var samples = new List<DetectionSample>();
            foreach (CocoImage image in document.Images)
            {
                string imagePath = ResolveCocoImagePath(root, split, image.FileName);
                List<GroundTruthDetection> groundTruths;
                if (!annotationsByImage.TryGetValue(image.Id, out groundTruths))
                    groundTruths = new List<GroundTruthDetection>();
                samples.Add(new DetectionSample(imagePath, groundTruths));
            }
            return new DetectionInput(samples, true, "coco", split,
                categoryNames.Values.Distinct(StringComparer.OrdinalIgnoreCase)
                    .OrderBy(value => value, StringComparer.OrdinalIgnoreCase).ToArray());
        }

        private static string ResolveCocoImagePath(string root, string split, string fileName)
        {
            if (string.IsNullOrWhiteSpace(fileName))
                throw new InvalidOperationException("COCO contains an image without file_name.");
            string[] directories = split == "val"
                ? new[] { "val", "valid", "val2017" }
                : new[] { split, split + "2017" };
            var candidates = new List<string> { Path.Combine(root, fileName) };
            foreach (string directory in directories)
                candidates.Add(Path.Combine(root, directory, fileName));
            candidates.Add(Path.Combine(root, "images", fileName));
            foreach (string candidate in candidates)
            {
                if (File.Exists(candidate))
                    return Path.GetFullPath(candidate);
            }

            string basename = Path.GetFileName(fileName);
            string discovered = Directory.EnumerateFiles(root, basename, SearchOption.AllDirectories)
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase).FirstOrDefault();
            if (discovered != null)
                return Path.GetFullPath(discovered);
            throw new FileNotFoundException("COCO image referenced by annotations was not found.", fileName);
        }

        private sealed class InputOptions
        {
            public bool Validate { get; set; }
            public bool ForceDataset { get; set; }
            public string Set { get; set; }
        }

        private sealed class ClassificationSample
        {
            public ClassificationSample(string path, string expectedClassName)
            {
                Path = path;
                ExpectedClassName = expectedClassName;
            }

            public string Path { get; private set; }
            public string ExpectedClassName { get; private set; }
        }

        private sealed class ClassificationInput
        {
            public ClassificationInput(List<ClassificationSample> samples, bool isDataset,
                string datasetFormat, string datasetSplit)
            {
                Samples = samples;
                IsDataset = isDataset;
                DatasetFormat = datasetFormat;
                DatasetSplit = datasetSplit;
            }

            public List<ClassificationSample> Samples { get; private set; }
            public bool IsDataset { get; private set; }
            public string DatasetFormat { get; private set; }
            public string DatasetSplit { get; private set; }
        }

        private sealed class DetectionSample
        {
            public DetectionSample(string path, List<GroundTruthDetection> groundTruths)
            {
                Path = path;
                GroundTruths = groundTruths;
            }

            public string Path { get; private set; }
            public List<GroundTruthDetection> GroundTruths { get; private set; }
        }

        private sealed class DetectionInput
        {
            public DetectionInput(List<DetectionSample> samples, bool isDataset,
                string datasetFormat, string datasetSplit, string[] datasetClassNames)
            {
                Samples = samples;
                IsDataset = isDataset;
                DatasetFormat = datasetFormat;
                DatasetSplit = datasetSplit;
                DatasetClassNames = datasetClassNames;
            }

            public List<DetectionSample> Samples { get; private set; }
            public bool IsDataset { get; private set; }
            public string DatasetFormat { get; private set; }
            public string DatasetSplit { get; private set; }
            public string[] DatasetClassNames { get; private set; }
        }

        private sealed class GroundTruthDetection
        {
            public GroundTruthDetection(string className, float x1, float y1, float x2, float y2)
            {
                ClassName = className;
                X1 = x1;
                Y1 = y1;
                X2 = x2;
                Y2 = y2;
            }

            public string ClassName { get; private set; }
            public float X1 { get; private set; }
            public float Y1 { get; private set; }
            public float X2 { get; private set; }
            public float Y2 { get; private set; }
        }

        [DataContract]
        private sealed class CocoDocument
        {
            [DataMember(Name = "images")]
            public List<CocoImage> Images { get; set; }

            [DataMember(Name = "annotations")]
            public List<CocoAnnotation> Annotations { get; set; }

            [DataMember(Name = "categories")]
            public List<CocoCategory> Categories { get; set; }
        }

        [DataContract]
        private sealed class CocoImage
        {
            [DataMember(Name = "id")]
            public long Id { get; set; }

            [DataMember(Name = "file_name")]
            public string FileName { get; set; }
        }

        [DataContract]
        private sealed class CocoAnnotation
        {
            [DataMember(Name = "image_id")]
            public long ImageId { get; set; }

            [DataMember(Name = "category_id")]
            public long CategoryId { get; set; }

            [DataMember(Name = "bbox")]
            public double[] BoundingBox { get; set; }
        }

        [DataContract]
        private sealed class CocoCategory
        {
            [DataMember(Name = "id")]
            public long Id { get; set; }

            [DataMember(Name = "name")]
            public string Name { get; set; }
        }

        private sealed class ClassificationValidationMetrics
        {
            private readonly string[] classNames;
            private readonly string datasetFormat;
            private readonly string datasetSplit;
            private readonly Dictionary<string, int> classIndices;
            private readonly int[] support;
            private readonly int[] truePositives;
            private readonly int[] falsePositives;
            private readonly int[] falseNegatives;
            private int correct;

            public ClassificationValidationMetrics(IReadOnlyList<string> classNames,
                string datasetFormat, string datasetSplit)
            {
                this.classNames = classNames.ToArray();
                this.datasetFormat = datasetFormat;
                this.datasetSplit = datasetSplit;
                classIndices = this.classNames.Select((name, index) => new { name, index })
                    .ToDictionary(item => item.name, item => item.index,
                        StringComparer.OrdinalIgnoreCase);
                support = new int[this.classNames.Length];
                truePositives = new int[this.classNames.Length];
                falsePositives = new int[this.classNames.Length];
                falseNegatives = new int[this.classNames.Length];
            }

            public void Add(string expectedClassName, OnnxClassification prediction)
            {
                int expectedIndex;
                if (!classIndices.TryGetValue(expectedClassName, out expectedIndex))
                {
                    throw new InvalidOperationException(string.Format(CultureInfo.InvariantCulture,
                        "Dataset class '{0}' is not present in the model class names.",
                        expectedClassName));
                }
                if (prediction.ClassIndex < 0 || prediction.ClassIndex >= classNames.Length)
                    throw new InvalidOperationException("The model returned an invalid classification index.");

                support[expectedIndex]++;
                if (expectedIndex == prediction.ClassIndex)
                {
                    correct++;
                    truePositives[expectedIndex]++;
                }
                else
                {
                    falseNegatives[expectedIndex]++;
                    falsePositives[prediction.ClassIndex]++;
                }
            }

            public Dictionary<string, object> ToReport()
            {
                int total = support.Sum();
                var perClass = new List<Dictionary<string, object>>();
                var precisions = new List<double>();
                var recalls = new List<double>();
                var f1Scores = new List<double>();
                for (int index = 0; index < classNames.Length; index++)
                {
                    if (support[index] == 0)
                        continue;
                    double precision = Divide(truePositives[index],
                        truePositives[index] + falsePositives[index]);
                    double recall = Divide(truePositives[index],
                        truePositives[index] + falseNegatives[index]);
                    double f1 = Divide(2.0 * precision * recall, precision + recall);
                    precisions.Add(precision);
                    recalls.Add(recall);
                    f1Scores.Add(f1);
                    perClass.Add(new Dictionary<string, object>
                    {
                        { "class_name", classNames[index] },
                        { "support", support[index] },
                        { "correct", truePositives[index] },
                        { "precision", precision },
                        { "recall", recall },
                        { "f1", f1 }
                    });
                }

                return new Dictionary<string, object>
                {
                    { "format", datasetFormat },
                    { "set", datasetSplit },
                    { "images", total },
                    { "correct", correct },
                    { "top1_accuracy", Divide(correct, total) },
                    { "macro_precision", Average(precisions) },
                    { "macro_recall", Average(recalls) },
                    { "macro_f1", Average(f1Scores) },
                    { "per_class", perClass }
                };
            }

            private static double Average(List<double> values)
            {
                return values.Count == 0 ? 0 : values.Average();
            }
        }

        private sealed class DetectionValidationMetrics
        {
            private readonly string[] classNames;
            private readonly string datasetFormat;
            private readonly string datasetSplit;
            private readonly Dictionary<string, int> classIndices;
            private readonly Dictionary<string, List<GroundTruthDetection>> groundTruths =
                new Dictionary<string, List<GroundTruthDetection>>(StringComparer.OrdinalIgnoreCase);
            private readonly List<PredictedDetection> predictions = new List<PredictedDetection>();

            public DetectionValidationMetrics(IReadOnlyList<string> classNames,
                string datasetFormat, string datasetSplit)
            {
                this.classNames = classNames.ToArray();
                this.datasetFormat = datasetFormat;
                this.datasetSplit = datasetSplit;
                classIndices = this.classNames.Select((name, index) => new { name, index })
                    .ToDictionary(item => item.name, item => item.index,
                        StringComparer.OrdinalIgnoreCase);
            }

            public void Add(string imagePath, List<GroundTruthDetection> imageGroundTruths,
                IReadOnlyList<OnnxDetection> imagePredictions)
            {
                if (groundTruths.ContainsKey(imagePath))
                    throw new InvalidOperationException("Duplicate image path in detection dataset.");
                var mappedGroundTruths = new List<GroundTruthDetection>();
                foreach (GroundTruthDetection groundTruth in imageGroundTruths)
                {
                    int classIndex;
                    if (!classIndices.TryGetValue(groundTruth.ClassName, out classIndex))
                    {
                        throw new InvalidOperationException(string.Format(CultureInfo.InvariantCulture,
                            "COCO class '{0}' is not present in the model class names.",
                            groundTruth.ClassName));
                    }
                    mappedGroundTruths.Add(new GroundTruthDetection(classIndex.ToString(CultureInfo.InvariantCulture),
                        groundTruth.X1, groundTruth.Y1, groundTruth.X2, groundTruth.Y2));
                }
                groundTruths.Add(imagePath, mappedGroundTruths);
                foreach (OnnxDetection prediction in imagePredictions)
                {
                    predictions.Add(new PredictedDetection(imagePath, prediction.ClassIndex,
                        prediction.Confidence, prediction.X1, prediction.Y1,
                        prediction.X2, prediction.Y2));
                }
            }

            public Dictionary<string, object> ToReport()
            {
                var perClass = new List<Dictionary<string, object>>();
                var ap50Values = new List<double>();
                var ap50To95Values = new List<double>();
                int truePositives = 0;
                int falsePositives = 0;
                int falseNegatives = 0;
                int groundTruthCount = groundTruths.Values.Sum(items => items.Count);
                int detectionCount = predictions.Count;

                for (int classIndex = 0; classIndex < classNames.Length; classIndex++)
                {
                    DetectionClassEvaluation at50 = EvaluateClass(classIndex, 0.50);
                    double ap50To95 = 0;
                    int thresholds = 0;
                    for (int step = 0; step < 10; step++)
                    {
                        DetectionClassEvaluation current = EvaluateClass(classIndex,
                            0.50 + step * 0.05);
                        if (!double.IsNaN(current.AveragePrecision))
                        {
                            ap50To95 += current.AveragePrecision;
                            thresholds++;
                        }
                    }
                    ap50To95 = thresholds == 0 ? double.NaN : ap50To95 / thresholds;
                    if (!double.IsNaN(at50.AveragePrecision))
                        ap50Values.Add(at50.AveragePrecision);
                    if (!double.IsNaN(ap50To95))
                        ap50To95Values.Add(ap50To95);
                    truePositives += at50.TruePositives;
                    falsePositives += at50.FalsePositives;
                    falseNegatives += at50.FalseNegatives;

                    if (at50.GroundTruthCount > 0 || at50.PredictionCount > 0)
                    {
                        perClass.Add(new Dictionary<string, object>
                        {
                            { "class_name", classNames[classIndex] },
                            { "ground_truth", at50.GroundTruthCount },
                            { "predictions", at50.PredictionCount },
                            { "true_positives", at50.TruePositives },
                            { "false_positives", at50.FalsePositives },
                            { "false_negatives", at50.FalseNegatives },
                            { "precision", at50.Precision },
                            { "recall", at50.Recall },
                            { "f1", at50.F1 },
                            { "ap50", at50.AveragePrecision },
                            { "ap50_95", ap50To95 }
                        });
                    }
                }

                double precision = Divide(truePositives, truePositives + falsePositives);
                double recall = Divide(truePositives, truePositives + falseNegatives);
                return new Dictionary<string, object>
                {
                    { "format", datasetFormat },
                    { "set", datasetSplit },
                    { "iou_matching", "0.50" },
                    { "images", groundTruths.Count },
                    { "ground_truth_boxes", groundTruthCount },
                    { "predictions", detectionCount },
                    { "true_positives", truePositives },
                    { "false_positives", falsePositives },
                    { "false_negatives", falseNegatives },
                    { "precision", precision },
                    { "recall", recall },
                    { "f1", Divide(2.0 * precision * recall, precision + recall) },
                    { "map50", Average(ap50Values) },
                    { "map50_95", Average(ap50To95Values) },
                    { "per_class", perClass }
                };
            }

            private DetectionClassEvaluation EvaluateClass(int classIndex, double iouThreshold)
            {
                var classGroundTruths = new Dictionary<string, List<GroundTruthDetection>>(
                    StringComparer.OrdinalIgnoreCase);
                int groundTruthCount = 0;
                foreach (KeyValuePair<string, List<GroundTruthDetection>> item in groundTruths)
                {
                    List<GroundTruthDetection> values = item.Value
                        .Where(groundTruth => GetClassIndex(groundTruth) == classIndex).ToList();
                    classGroundTruths[item.Key] = values;
                    groundTruthCount += values.Count;
                }

                List<PredictedDetection> classPredictions = predictions
                    .Where(prediction => prediction.ClassIndex == classIndex)
                    .OrderByDescending(prediction => prediction.Confidence)
                    .ToList();
                var matched = new Dictionary<string, bool[]>(StringComparer.OrdinalIgnoreCase);
                foreach (KeyValuePair<string, List<GroundTruthDetection>> item in classGroundTruths)
                    matched[item.Key] = new bool[item.Value.Count];

                int truePositives = 0;
                int falsePositives = 0;
                var truePositiveFlags = new List<bool>();
                foreach (PredictedDetection prediction in classPredictions)
                {
                    List<GroundTruthDetection> candidates;
                    if (!classGroundTruths.TryGetValue(prediction.ImagePath, out candidates))
                        candidates = new List<GroundTruthDetection>();
                    bool[] matchedCandidates = matched[prediction.ImagePath];
                    int bestIndex = -1;
                    float bestIou = 0;
                    for (int index = 0; index < candidates.Count; index++)
                    {
                        if (matchedCandidates[index])
                            continue;
                        float currentIou = IntersectionOverUnion(prediction, candidates[index]);
                        if (currentIou >= iouThreshold && currentIou > bestIou)
                        {
                            bestIou = currentIou;
                            bestIndex = index;
                        }
                    }
                    if (bestIndex >= 0)
                    {
                        matchedCandidates[bestIndex] = true;
                        truePositives++;
                        truePositiveFlags.Add(true);
                    }
                    else
                    {
                        falsePositives++;
                        truePositiveFlags.Add(false);
                    }
                }

                int falseNegatives = groundTruthCount - truePositives;
                double precision = Divide(truePositives, truePositives + falsePositives);
                double recall = Divide(truePositives, truePositives + falseNegatives);
                return new DetectionClassEvaluation(groundTruthCount, classPredictions.Count,
                    truePositives, falsePositives, falseNegatives, precision, recall,
                    Divide(2.0 * precision * recall, precision + recall),
                    CalculateAveragePrecision(truePositiveFlags, groundTruthCount));
            }

            private int GetClassIndex(GroundTruthDetection groundTruth)
            {
                int classIndex;
                return int.TryParse(groundTruth.ClassName, NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out classIndex) ? classIndex : -1;
            }

            private static double CalculateAveragePrecision(List<bool> truePositiveFlags,
                int groundTruthCount)
            {
                if (groundTruthCount == 0)
                    return double.NaN;
                int truePositives = 0;
                int falsePositives = 0;
                var precisions = new List<double>();
                var recalls = new List<double>();
                foreach (bool truePositive in truePositiveFlags)
                {
                    if (truePositive)
                        truePositives++;
                    else
                        falsePositives++;
                    precisions.Add(Divide(truePositives, truePositives + falsePositives));
                    recalls.Add(Divide(truePositives, groundTruthCount));
                }

                double sum = 0;
                for (int step = 0; step <= 100; step++)
                {
                    double threshold = step / 100.0;
                    double maximum = 0;
                    for (int index = 0; index < recalls.Count; index++)
                    {
                        if (recalls[index] >= threshold)
                            maximum = Math.Max(maximum, precisions[index]);
                    }
                    sum += maximum;
                }
                return sum / 101.0;
            }

            private static float IntersectionOverUnion(PredictedDetection prediction,
                GroundTruthDetection groundTruth)
            {
                float x1 = Math.Max(prediction.X1, groundTruth.X1);
                float y1 = Math.Max(prediction.Y1, groundTruth.Y1);
                float x2 = Math.Min(prediction.X2, groundTruth.X2);
                float y2 = Math.Min(prediction.Y2, groundTruth.Y2);
                float intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
                float predictionArea = Math.Max(0, prediction.X2 - prediction.X1) *
                    Math.Max(0, prediction.Y2 - prediction.Y1);
                float groundTruthArea = Math.Max(0, groundTruth.X2 - groundTruth.X1) *
                    Math.Max(0, groundTruth.Y2 - groundTruth.Y1);
                float union = predictionArea + groundTruthArea - intersection;
                return union <= 0 ? 0 : intersection / union;
            }

            private static double Average(List<double> values)
            {
                return values.Count == 0 ? 0 : values.Average();
            }
        }

        private sealed class PredictedDetection
        {
            public PredictedDetection(string imagePath, int classIndex, float confidence,
                float x1, float y1, float x2, float y2)
            {
                ImagePath = imagePath;
                ClassIndex = classIndex;
                Confidence = confidence;
                X1 = x1;
                Y1 = y1;
                X2 = x2;
                Y2 = y2;
            }

            public string ImagePath { get; private set; }
            public int ClassIndex { get; private set; }
            public float Confidence { get; private set; }
            public float X1 { get; private set; }
            public float Y1 { get; private set; }
            public float X2 { get; private set; }
            public float Y2 { get; private set; }
        }

        private sealed class DetectionClassEvaluation
        {
            public DetectionClassEvaluation(int groundTruthCount, int predictionCount,
                int truePositives, int falsePositives, int falseNegatives,
                double precision, double recall, double f1, double averagePrecision)
            {
                GroundTruthCount = groundTruthCount;
                PredictionCount = predictionCount;
                TruePositives = truePositives;
                FalsePositives = falsePositives;
                FalseNegatives = falseNegatives;
                Precision = precision;
                Recall = recall;
                F1 = f1;
                AveragePrecision = averagePrecision;
            }

            public int GroundTruthCount { get; private set; }
            public int PredictionCount { get; private set; }
            public int TruePositives { get; private set; }
            public int FalsePositives { get; private set; }
            public int FalseNegatives { get; private set; }
            public double Precision { get; private set; }
            public double Recall { get; private set; }
            public double F1 { get; private set; }
            public double AveragePrecision { get; private set; }
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
            Console.WriteLine("  OnnxVisionCLI.exe <model.onnx> <image-file|image-directory|dataset> [task options]");
            Console.WriteLine("  Classification options: [provider] [repeats] [roi-x roi-y roi-width roi-height]");
            Console.WriteLine("  Detection options: [confidence] [repeats] [provider]");
            Console.WriteLine("  Dataset options: [-dataset] [-validate] [-set train|val|test] [--json]");
            Console.WriteLine("  ImageNet classification datasets use train/val/test/<class>/image files.");
            Console.WriteLine("  COCO detection datasets use annotations/instances_<set>.json or " +
                "<set>/_annotations.coco.json.");
            Console.WriteLine("  Validation is available only when a labeled dataset is supplied; " +
                "default dataset set is val when present.");
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
