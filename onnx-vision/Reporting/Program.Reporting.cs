using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using OnnxVision.Classification;
using OnnxVision.Detection;
using OnnxVision.Runtime;

namespace OnnxVision
{
    internal static partial class Program
    {
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
            RoiPlacement roi, OnnxExecutionProvider[] providers, int batchSize,
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
            Console.WriteLine("Batch size: {0} ({1})", batchSize,
                classifier.SupportsDynamicBatch ? "dynamic" : "fixed");
        }

        private static void PrintDetectionInformation(
            OnnxObjectDetectionModel detector, OnnxExecutionProvider[] providers, int batchSize,
            bool isDataset, string datasetFormat, string datasetSplit)
        {
            Console.WriteLine("Detection contract: {0} {1}",
                OnnxVisionContract.ObjectDetectionName, OnnxVisionContract.Version);
            Console.WriteLine("Provider: {0}", detector.ActualProvider);
            Console.WriteLine("Requested providers: {0}", FormatProviders(providers));
            Console.WriteLine("Input contract: " + detector.InputDescription);
            Console.WriteLine("Batch size: {0} ({1})", batchSize,
                detector.SupportsDynamicBatch ? "dynamic" : "fixed");
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
            Console.WriteLine("  Dynamic-batch models: [-batch-size N] (default: 1)");
            Console.WriteLine("  Fixed-batch models always use the batch size declared by the model.");
            Console.WriteLine("  Dataset options: [-dataset] [-validate] [-set train|val|test] [--json]");
            Console.WriteLine("  ImageNet classification datasets use train/val/test/<class>/image files.");
            Console.WriteLine("  COCO detection datasets use annotations/instances_<set>.json or " +
                "<set>/_annotations.coco.json.");
            Console.WriteLine("  Validation is available only when a labeled dataset is supplied; " +
                "default dataset set is val when present.");
            Console.WriteLine("  'detect' remains an optional object-detection command alias.");
            Console.WriteLine("Providers: cpu");
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
    }
}
