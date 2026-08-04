using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using OnnxVision.Classification;
using OnnxVision.Detection;
using OnnxVision.Runtime;

namespace OnnxVision
{
    internal static partial class Program
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

    }
}
