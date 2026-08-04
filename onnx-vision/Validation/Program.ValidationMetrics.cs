using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using OnnxVision.Classification;
using OnnxVision.Detection;

namespace OnnxVision
{
    internal static partial class Program
    {
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
    }
}
