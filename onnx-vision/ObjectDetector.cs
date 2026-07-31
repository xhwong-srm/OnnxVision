using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;
using System.Text.RegularExpressions;
using Euresys.Open_eVision_22_12;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxVision
{
    /// <summary>
    /// Model-neutral object detector for the onnx-vision-detection-v1 contract.
    /// Keep one instance alive and reuse it. Instances are not thread-safe.
    /// </summary>
    public sealed class ObjectDetector : IDisposable
    {
        private const string Contract = "onnx-vision-detection-v1";
        private readonly InferenceSession session;
        private readonly string[] classNames;
        private readonly InputContract inputContract;
        private readonly bool nmsRequired;
        private readonly EROIBW8 attachedBw8Roi;
        private readonly EROIC24 attachedC24Roi;
        private byte[] byteTensorBuffer;
        private int byteTensorWidth;
        private int byteTensorHeight;
        private IReadOnlyCollection<NamedOnnxValue> inputs;
        private long preprocessTicks;
        private long inferenceTicks;
        private bool disposed;

        public ObjectDetector(string modelPath, IEnumerable<string> classNames = null,
            ExecutionProvider executionProvider = ExecutionProvider.Cpu)
        {
            if (string.IsNullOrWhiteSpace(modelPath))
                throw new ArgumentException("A model path is required.", "modelPath");
            ExecutionProvider = executionProvider;
            using (var options = CreateSessionOptions(executionProvider))
                session = new InferenceSession(modelPath, options);

            string task;
            string contract;
            var metadata = session.ModelMetadata.CustomMetadataMap;
            if (!metadata.TryGetValue("vision_task", out task) || task != "object_detection" ||
                !metadata.TryGetValue("detection_contract", out contract) || contract != Contract)
                throw new NotSupportedException("Expected the " + Contract + " object-detection metadata contract.");
            string nmsRequiredValue;
            if (!metadata.TryGetValue("nms_required", out nmsRequiredValue) ||
                !bool.TryParse(nmsRequiredValue, out nmsRequired))
                throw new NotSupportedException(
                    "Expected boolean nms_required metadata in the object-detection model.");
            RequireOutput("boxes", typeof(float), 3);
            RequireOutput("scores", typeof(float), 2);
            RequireOutput("class_ids", typeof(long), 2);

            var embeddedClassNames = ReadEmbeddedClassNames(metadata);
            var explicitClassNames = classNames == null ? null : classNames.ToArray();
            ValidateClassNames(explicitClassNames, "classNames");
            ValidateClassNames(embeddedClassNames, "model metadata");
            if (explicitClassNames != null && embeddedClassNames != null &&
                !explicitClassNames.SequenceEqual(embeddedClassNames, StringComparer.Ordinal))
                throw new InvalidOperationException("Explicit class names do not match the model metadata.");
            this.classNames = explicitClassNames ?? embeddedClassNames;
            if (this.classNames == null)
                throw new ArgumentException("Class names must be supplied or embedded in model metadata.", "classNames");

            var inputMetadata = session.InputMetadata.Single();
            InputName = inputMetadata.Key;
            var dimensions = inputMetadata.Value.Dimensions;
            if (inputMetadata.Value.ElementType == typeof(byte) && dimensions.Length == 4 && dimensions[1] == 1)
                inputContract = InputContract.Bw8Nchw;
            else if (inputMetadata.Value.ElementType == typeof(byte) && dimensions.Length == 4 && dimensions[3] == 3)
                inputContract = InputContract.C24Nhwc;
            else
                throw new NotSupportedException(
                    "Expected uint8 BW8 NCHW or uint8 C24 NHWC input with embedded preprocessing.");
            if (inputContract == InputContract.C24Nhwc)
                attachedC24Roi = new EROIC24();
            else
                attachedBw8Roi = new EROIBW8();
        }

        public string InputName { get; private set; }
        public ExecutionProvider ExecutionProvider { get; private set; }
        public bool NmsRequired { get { return nmsRequired; } }
        public double PreprocessMilliseconds { get { return preprocessTicks * 1000.0 / Stopwatch.Frequency; } }
        public double InferenceMilliseconds { get { return inferenceTicks * 1000.0 / Stopwatch.Frequency; } }
        public IReadOnlyList<string> ClassNames { get { return classNames; } }

        public string InputDescription
        {
            get
            {
                return inputContract == InputContract.C24Nhwc
                    ? "uint8 [1,H,W,3] C24 NHWC BGR; preprocessing runs inside ONNX"
                    : "uint8 [1,1,H,W] BW8 NCHW; preprocessing runs inside ONNX";
            }
        }

        public IReadOnlyList<Detection> Detect(string imagePath, float confidenceThreshold = 0.5f,
            float nmsIouThreshold = 1.0f)
        {
            ThrowIfDisposed();
            if (string.IsNullOrWhiteSpace(imagePath))
                throw new ArgumentException("An image path is required.", "imagePath");
            if (inputContract == InputContract.C24Nhwc)
            {
                using (var image = new EImageC24())
                {
                    image.Load(imagePath);
                    return Detect(image, confidenceThreshold, nmsIouThreshold);
                }
            }
            using (var image = new EImageBW8())
            {
                image.Load(imagePath);
                return Detect(image, confidenceThreshold, nmsIouThreshold);
            }
        }

        public IReadOnlyList<Detection> Detect(EImageBW8 image, float confidenceThreshold = 0.5f,
            float nmsIouThreshold = 1.0f)
        {
            if (image == null) throw new ArgumentNullException("image");
            return Detect(image, new RoiPlacement(0, 0, image.Width, image.Height),
                confidenceThreshold, nmsIouThreshold);
        }

        public IReadOnlyList<Detection> Detect(EImageBW8 image, RoiPlacement placement,
            float confidenceThreshold = 0.5f, float nmsIouThreshold = 1.0f)
        {
            ThrowIfDisposed();
            if (image == null) throw new ArgumentNullException("image");
            EnsureBw8Model();
            ValidatePlacement(placement, image.Width, image.Height);
            attachedBw8Roi.Detach();
            attachedBw8Roi.Attach(image);
            attachedBw8Roi.SetPlacement(placement.X, placement.Y, placement.Width, placement.Height);
            try
            {
                return Detect(attachedBw8Roi, confidenceThreshold, nmsIouThreshold);
            }
            finally
            {
                attachedBw8Roi.Detach();
            }
        }

        public IReadOnlyList<Detection> Detect(EImageC24 image, float confidenceThreshold = 0.5f,
            float nmsIouThreshold = 1.0f)
        {
            if (image == null) throw new ArgumentNullException("image");
            return Detect(image, new RoiPlacement(0, 0, image.Width, image.Height),
                confidenceThreshold, nmsIouThreshold);
        }

        public IReadOnlyList<Detection> Detect(EImageC24 image, RoiPlacement placement,
            float confidenceThreshold = 0.5f, float nmsIouThreshold = 1.0f)
        {
            ThrowIfDisposed();
            if (image == null) throw new ArgumentNullException("image");
            EnsureC24Model();
            ValidatePlacement(placement, image.Width, image.Height);
            attachedC24Roi.Detach();
            attachedC24Roi.Attach(image);
            attachedC24Roi.SetPlacement(placement.X, placement.Y, placement.Width, placement.Height);
            try
            {
                return Detect(attachedC24Roi, confidenceThreshold, nmsIouThreshold);
            }
            finally
            {
                attachedC24Roi.Detach();
            }
        }

        public IReadOnlyList<Detection> Detect(EROIBW8 roi, float confidenceThreshold = 0.5f,
            float nmsIouThreshold = 1.0f)
        {
            ThrowIfDisposed();
            if (roi == null) throw new ArgumentNullException("roi");
            EnsureBw8Model();
            ValidateThresholds(confidenceThreshold, nmsIouThreshold);
            var started = Stopwatch.GetTimestamp();
            EnsureByteTensor(roi.Width, roi.Height, 1);
            for (var y = 0; y < roi.Height; y++)
                Marshal.Copy(roi.GetImagePtr(0, y), byteTensorBuffer, y * roi.Width, roi.Width);
            return RunInference(roi.Width, roi.Height, confidenceThreshold, nmsIouThreshold, started);
        }

        public IReadOnlyList<Detection> Detect(EROIC24 roi, float confidenceThreshold = 0.5f,
            float nmsIouThreshold = 1.0f)
        {
            ThrowIfDisposed();
            if (roi == null) throw new ArgumentNullException("roi");
            EnsureC24Model();
            ValidateThresholds(confidenceThreshold, nmsIouThreshold);
            var started = Stopwatch.GetTimestamp();
            var rowBytes = checked(roi.Width * 3);
            EnsureByteTensor(roi.Width, roi.Height, 3);
            for (var y = 0; y < roi.Height; y++)
                Marshal.Copy(roi.GetImagePtr(0, y), byteTensorBuffer, y * rowBytes, rowBytes);
            return RunInference(roi.Width, roi.Height, confidenceThreshold, nmsIouThreshold, started);
        }

        public void ResetTimings()
        {
            ThrowIfDisposed();
            preprocessTicks = 0;
            inferenceTicks = 0;
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            if (attachedBw8Roi != null) attachedBw8Roi.Dispose();
            if (attachedC24Roi != null) attachedC24Roi.Dispose();
            session.Dispose();
        }

        private IReadOnlyList<Detection> RunInference(int width, int height, float threshold,
            float nmsIouThreshold, long started)
        {
            var preprocessed = Stopwatch.GetTimestamp();
            var candidates = new List<Detection>();
            using (var results = session.Run(inputs))
            {
                var boxes = results.Single(item => item.Name == "boxes").AsTensor<float>();
                var scores = results.Single(item => item.Name == "scores").AsTensor<float>();
                var classIds = results.Single(item => item.Name == "class_ids").AsTensor<long>();
                if (boxes.Length != scores.Length * 4 || classIds.Length != scores.Length)
                    throw new InvalidOperationException("Detection output tensor sizes do not agree.");
                for (var index = 0; index < scores.Length; index++)
                {
                    var score = scores.GetValue(index);
                    if (score < threshold) continue;
                    var classIndex = checked((int)classIds.GetValue(index));
                    if (classIndex < 0 || classIndex >= classNames.Length)
                        throw new InvalidOperationException("Model returned an out-of-range class ID.");
                    candidates.Add(new Detection(
                        classNames[classIndex], classIndex, score,
                        Clamp(boxes.GetValue(index * 4) * width, 0, width),
                        Clamp(boxes.GetValue(index * 4 + 1) * height, 0, height),
                        Clamp(boxes.GetValue(index * 4 + 2) * width, 0, width),
                        Clamp(boxes.GetValue(index * 4 + 3) * height, 0, height)));
                }
            }
            var inferred = Stopwatch.GetTimestamp();
            preprocessTicks += preprocessed - started;
            inferenceTicks += inferred - preprocessed;
            return !nmsRequired || nmsIouThreshold >= 1.0f
                ? candidates
                : ApplyClassAwareNms(candidates, nmsIouThreshold);
        }

        private static IReadOnlyList<Detection> ApplyClassAwareNms(List<Detection> candidates, float threshold)
        {
            var kept = new List<Detection>();
            foreach (var candidate in candidates.OrderByDescending(item => item.Confidence))
            {
                if (kept.Any(item => item.ClassIndex == candidate.ClassIndex && IntersectionOverUnion(item, candidate) > threshold))
                    continue;
                kept.Add(candidate);
            }
            return kept;
        }

        private static float IntersectionOverUnion(Detection a, Detection b)
        {
            var intersectionWidth = Math.Max(0, Math.Min(a.X2, b.X2) - Math.Max(a.X1, b.X1));
            var intersectionHeight = Math.Max(0, Math.Min(a.Y2, b.Y2) - Math.Max(a.Y1, b.Y1));
            var intersection = intersectionWidth * intersectionHeight;
            var union = a.Width * a.Height + b.Width * b.Height - intersection;
            return union <= 0 ? 0 : intersection / union;
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
            inputs = new[] { NamedOnnxValue.CreateFromTensor(InputName, new DenseTensor<byte>(byteTensorBuffer, shape)) };
        }

        private void RequireOutput(string name, Type elementType, int rank)
        {
            NodeMetadata metadata;
            if (!session.OutputMetadata.TryGetValue(name, out metadata) ||
                metadata.ElementType != elementType || metadata.Dimensions.Length != rank)
                throw new NotSupportedException(string.Format(
                    "Expected output {0} with element type {1} and rank {2}.", name, elementType.Name, rank));
        }

        private static SessionOptions CreateSessionOptions(ExecutionProvider provider)
        {
            var options = new SessionOptions();
            if (provider == ExecutionProvider.DirectML)
            {
                options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                options.EnableMemoryPattern = false;
                options.AppendExecutionProvider_DML(0);
            }
            else if (provider == ExecutionProvider.OpenVinoCpu)
                options.AppendExecutionProvider_OpenVINO("CPU");
            else if (provider == ExecutionProvider.OpenVinoGpu)
                options.AppendExecutionProvider_OpenVINO("GPU");
            return options;
        }

        private static string[] ReadEmbeddedClassNames(IReadOnlyDictionary<string, string> metadata)
        {
            string serialized;
            if (!metadata.TryGetValue("names", out serialized) || string.IsNullOrWhiteSpace(serialized))
                return null;
            try
            {
                var serializer = new DataContractJsonSerializer(
                    typeof(Dictionary<string, string>),
                    new DataContractJsonSerializerSettings { UseSimpleDictionaryFormat = true });
                using (var stream = new MemoryStream(Encoding.UTF8.GetBytes(serialized)))
                    return BuildClassNames((Dictionary<string, string>)serializer.ReadObject(stream));
            }
            catch (SerializationException)
            {
                var matches = Regex.Matches(serialized, "(?<index>\\d+)\\s*:\\s*['\"](?<name>[^'\"]*)['\"]");
                if (matches.Count == 0)
                    throw new InvalidOperationException("The names metadata is not a valid class mapping.");
                var values = new Dictionary<string, string>();
                foreach (Match match in matches)
                    values.Add(match.Groups["index"].Value, match.Groups["name"].Value);
                return BuildClassNames(values);
            }
        }

        private static string[] BuildClassNames(IDictionary<string, string> mapping)
        {
            if (mapping == null || mapping.Count == 0)
                throw new InvalidOperationException("The names metadata must contain at least one class.");
            var names = new string[mapping.Count];
            foreach (var item in mapping)
            {
                int index;
                if (!int.TryParse(item.Key, out index) || index < 0 || index >= names.Length || names[index] != null)
                    throw new InvalidOperationException("Class indices must be unique and contiguous from zero.");
                names[index] = item.Value;
            }
            ValidateClassNames(names, "model metadata");
            return names;
        }

        private static void ValidateClassNames(string[] names, string source)
        {
            if (names != null && (names.Length == 0 || names.Any(string.IsNullOrWhiteSpace)))
                throw new ArgumentException("Class names from " + source + " must be non-empty.");
        }

        private static void ValidateThresholds(float confidence, float nms)
        {
            if (confidence < 0 || confidence > 1)
                throw new ArgumentOutOfRangeException("confidenceThreshold");
            if (nms < 0 || nms > 1)
                throw new ArgumentOutOfRangeException("nmsIouThreshold");
        }

        private static void ValidatePlacement(RoiPlacement placement, int imageWidth, int imageHeight)
        {
            if (placement == null) throw new ArgumentNullException("placement");
            if (placement.X < 0 || placement.Y < 0 ||
                placement.X + placement.Width > imageWidth || placement.Y + placement.Height > imageHeight)
                throw new ArgumentOutOfRangeException("placement");
        }

        private void EnsureBw8Model()
        {
            if (inputContract != InputContract.Bw8Nchw)
                throw new InvalidOperationException("This model requires EImageC24 or EROIC24 input.");
        }

        private void EnsureC24Model()
        {
            if (inputContract != InputContract.C24Nhwc)
                throw new InvalidOperationException("This model requires EImageBW8 or EROIBW8 input.");
        }

        private static float Clamp(float value, float minimum, float maximum)
        {
            return Math.Max(minimum, Math.Min(maximum, value));
        }

        private void ThrowIfDisposed()
        {
            if (disposed) throw new ObjectDisposedException("ObjectDetector");
        }

        private enum InputContract
        {
            Bw8Nchw,
            C24Nhwc
        }
    }

    public sealed class Detection
    {
        public Detection(string name, int classIndex, float confidence, float x1, float y1, float x2, float y2)
        {
            Name = name;
            ClassIndex = classIndex;
            Confidence = confidence;
            X1 = x1;
            Y1 = y1;
            X2 = x2;
            Y2 = y2;
        }

        public string Name { get; private set; }
        public int ClassIndex { get; private set; }
        public float Confidence { get; private set; }
        public float X1 { get; private set; }
        public float Y1 { get; private set; }
        public float X2 { get; private set; }
        public float Y2 { get; private set; }
        public float Width { get { return Math.Max(0, X2 - X1); } }
        public float Height { get { return Math.Max(0, Y2 - Y1); } }
    }
}
