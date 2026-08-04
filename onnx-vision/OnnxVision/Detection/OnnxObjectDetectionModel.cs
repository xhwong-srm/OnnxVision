using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxVision.Imaging;
using OnnxVision.Runtime;

namespace OnnxVision.Detection
{
    /// <summary>
    /// Model-neutral object detector for the onnx-vision-detection-v1 contract.
    /// A model instance can be reused and supports concurrent predictions.
    /// </summary>
    public sealed class OnnxObjectDetectionModel : IDisposable
    {
        private const string Contract = "onnx-vision-detection-v1";
        private const string BoxesOutputName = "boxes";
        private const string ScoresOutputName = "scores";
        private const string ClassIdsOutputName = "class_ids";
        private readonly InferenceSession session;
        private readonly string[] classNames;
        private readonly OnnxPixelFormat requiredPixelFormat;
        private readonly int inputWidth;
        private readonly int inputHeight;
        private readonly bool nmsRequired;
        private readonly string[] inputNames;
        private readonly string[] outputNames;
        private readonly ConcurrentBag<InputWorkspace> workspaces = new ConcurrentBag<InputWorkspace>();
        private bool disposed;

        public OnnxObjectDetectionModel(string modelPath, params OnnxExecutionProvider[] providerPriority)
            : this(modelPath, null, providerPriority)
        {
        }

        public OnnxObjectDetectionModel(string modelPath, IEnumerable<string> classNames,
            params OnnxExecutionProvider[] providerPriority)
        {
            if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
                throw new FileNotFoundException("ONNX model file was not found.", modelPath);

            if (providerPriority == null || providerPriority.Length == 0)
                providerPriority = new[] { OnnxExecutionProvider.Cpu };

            var providers = ValidateProviders(providerPriority);
            RequestedProviders = providers;
            session = CreateSession(modelPath, providers, out OnnxExecutionProvider actualProvider);
            ActualProvider = actualProvider;

            try
            {
                nmsRequired = ValidateContract();

                var explicitClassNames = classNames == null ? null : classNames.ToArray();
                var embeddedClassNames = ReadEmbeddedClassNames(session.ModelMetadata.CustomMetadataMap);
                ValidateClassNames(explicitClassNames, "classNames");
                ValidateClassNames(embeddedClassNames, "model metadata");
                if (explicitClassNames != null && embeddedClassNames != null &&
                    !explicitClassNames.SequenceEqual(embeddedClassNames, StringComparer.Ordinal))
                {
                    throw new InvalidOperationException("Explicit class names do not match the model metadata.");
                }

                this.classNames = explicitClassNames ?? embeddedClassNames;
                if (this.classNames == null)
                    throw new ArgumentException(
                        "Class names must be supplied or embedded in model metadata.", "classNames");

                var input = session.InputMetadata.Single();
                inputNames = new[] { input.Key };
                var dimensions = input.Value.Dimensions;
                if (input.Value.ElementType == typeof(byte) && dimensions.Length == 4 && dimensions[1] == 1)
                {
                    requiredPixelFormat = OnnxPixelFormat.Bw8;
                    inputHeight = dimensions[2];
                    inputWidth = dimensions[3];
                }
                else if (input.Value.ElementType == typeof(byte) && dimensions.Length == 4 && dimensions[3] == 3)
                {
                    requiredPixelFormat = OnnxPixelFormat.Bgr24;
                    inputHeight = dimensions[1];
                    inputWidth = dimensions[2];
                }
                else
                {
                    throw new NotSupportedException(
                        "ONNX object detection requires embedded preprocessing with uint8 [1,1,H,W] BW8 NCHW or uint8 [1,H,W,3] C24 NHWC BGR input.");
                }
            }
            catch
            {
                session.Dispose();
                throw;
            }

            outputNames = new[] { BoxesOutputName, ScoresOutputName, ClassIdsOutputName };
        }

        public IReadOnlyList<OnnxExecutionProvider> RequestedProviders { get; private set; }
        public OnnxExecutionProvider ActualProvider { get; private set; }
        public IReadOnlyList<string> ClassNames { get { return classNames; } }
        public OnnxPixelFormat RequiredPixelFormat { get { return requiredPixelFormat; } }
        public int InputWidth { get { return inputWidth; } }
        public int InputHeight { get { return inputHeight; } }
        public bool RequiresColorInput { get { return requiredPixelFormat == OnnxPixelFormat.Bgr24; } }
        public bool NmsRequired { get { return nmsRequired; } }

        public string InputDescription
        {
            get
            {
                return RequiresColorInput
                    ? "uint8 [1,H,W,3] C24 NHWC BGR; preprocessing runs inside ONNX"
                    : "uint8 [1,1,H,W] BW8 NCHW; preprocessing runs inside ONNX";
            }
        }

        public IReadOnlyList<OnnxDetection> Detect(OnnxImageBuffer image,
            float confidenceThreshold = 0.5f, float nmsIouThreshold = 1.0f)
        {
            ThrowIfDisposed();
            ValidateImage(image);
            ValidateThresholds(confidenceThreshold, nmsIouThreshold);

            InputWorkspace workspace;
            if (!workspaces.TryTake(out workspace))
                workspace = new InputWorkspace();

            try
            {
                int channels = image.BytesPerPixel;
                int rowBytes = checked(image.Width * channels);
                int requiredLength = checked(rowBytes * image.Height);
                long[] inputShape = requiredPixelFormat == OnnxPixelFormat.Bw8
                    ? new long[] { 1, 1, image.Height, image.Width }
                    : new long[] { 1, image.Height, image.Width, 3 };

                OrtValue inputValue;
                if (image.RowStride == rowBytes)
                {
                    inputValue = workspace.PrepareDirectInput(image.Data, requiredLength, inputShape);
                }
                else
                {
                    IntPtr destination = workspace.PrepareStagingInput(requiredLength, inputShape);
                    CopyRows(image.Data, image.RowStride, destination, rowBytes, image.Height);
                    inputValue = workspace.StagingInput;
                }
                workspace.InputValues[0] = inputValue;

                using (var results = session.Run(workspace.RunOptions, inputNames,
                    workspace.InputValues, outputNames))
                {
                    var outputs = results.ToArray();
                    var boxes = outputs[0].GetTensorDataAsSpan<float>();
                    var scores = outputs[1].GetTensorDataAsSpan<float>();
                    var classIds = outputs[2].GetTensorDataAsSpan<long>();
                    if (boxes.Length != scores.Length * 4 || classIds.Length != scores.Length)
                        throw new InvalidOperationException("Detection output tensor sizes do not agree.");

                    var candidates = new List<OnnxDetection>();
                    for (int index = 0; index < scores.Length; index++)
                    {
                        float score = scores[index];
                        if (score < confidenceThreshold)
                            continue;

                        long classId = classIds[index];
                        if (classId < 0 || classId >= classNames.Length)
                            throw new InvalidOperationException("Model returned an out-of-range class ID.");
                        int classIndex = (int)classId;

                        candidates.Add(new OnnxDetection(
                            classNames[classIndex], classIndex, score,
                            Clamp(boxes[index * 4] * image.Width, 0, image.Width),
                            Clamp(boxes[index * 4 + 1] * image.Height, 0, image.Height),
                            Clamp(boxes[index * 4 + 2] * image.Width, 0, image.Width),
                            Clamp(boxes[index * 4 + 3] * image.Height, 0, image.Height)));
                    }

                    if (!nmsRequired || nmsIouThreshold >= 1.0f)
                        return candidates;
                    return ApplyClassAwareNms(candidates, nmsIouThreshold);
                }
            }
            finally
            {
                if (disposed)
                    workspace.Dispose();
                else
                    workspaces.Add(workspace);
            }
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;

            InputWorkspace workspace;
            while (workspaces.TryTake(out workspace))
                workspace.Dispose();
            session.Dispose();
        }

        private bool ValidateContract()
        {
            var metadata = session.ModelMetadata.CustomMetadataMap;
            string task;
            string contract;
            if (!metadata.TryGetValue("vision_task", out task) || task != "object_detection" ||
                !metadata.TryGetValue("detection_contract", out contract) || contract != Contract)
            {
                throw new NotSupportedException(
                    "Expected the " + Contract + " object-detection metadata contract.");
            }

            string nmsRequiredValue;
            if (!metadata.TryGetValue("nms_required", out nmsRequiredValue) ||
                !bool.TryParse(nmsRequiredValue, out bool parsedNmsRequired))
            {
                throw new NotSupportedException(
                    "Expected boolean nms_required metadata in the object-detection model.");
            }

            RequireOutput(BoxesOutputName, typeof(float), 3);
            RequireOutput(ScoresOutputName, typeof(float), 2);
            RequireOutput(ClassIdsOutputName, typeof(long), 2);
            return parsedNmsRequired;
        }

        private void RequireOutput(string name, Type elementType, int rank)
        {
            NodeMetadata metadata;
            if (!session.OutputMetadata.TryGetValue(name, out metadata) ||
                metadata.ElementType != elementType || metadata.Dimensions.Length != rank)
            {
                throw new NotSupportedException(string.Format(
                    "Expected output {0} with element type {1} and rank {2}.",
                    name, elementType.Name, rank));
            }
        }

        private void ValidateImage(OnnxImageBuffer image)
        {
            if (image.Data == IntPtr.Zero)
                throw new ArgumentException("The ONNX source image has no accessible pixel buffer.", "image");
            if (image.Width <= 0 || image.Height <= 0)
                throw new ArgumentException("The ONNX source image must have positive dimensions.", "image");
            if (image.PixelFormat != requiredPixelFormat)
            {
                throw new ArgumentException(string.Format(
                    "The ONNX model requires {0} input, but the source image is {1}.",
                    requiredPixelFormat, image.PixelFormat), "image");
            }
            if (inputWidth > 0 && image.Width != inputWidth)
            {
                throw new ArgumentException(string.Format(
                    "The ONNX model requires input width {0}, but the selected image region is {1}.",
                    inputWidth, image.Width), "image");
            }
            if (inputHeight > 0 && image.Height != inputHeight)
            {
                throw new ArgumentException(string.Format(
                    "The ONNX model requires input height {0}, but the selected image region is {1}.",
                    inputHeight, image.Height), "image");
            }

            int rowBytes = checked(image.Width * image.BytesPerPixel);
            if (Math.Abs((long)image.RowStride) < rowBytes)
                throw new ArgumentException("The ONNX source row stride is smaller than one image row.", "image");
        }

        private static OnnxExecutionProvider[] ValidateProviders(OnnxExecutionProvider[] providers)
        {
            var uniqueProviders = new List<OnnxExecutionProvider>();
            foreach (OnnxExecutionProvider provider in providers)
            {
                if (!Enum.IsDefined(typeof(OnnxExecutionProvider), provider))
                    throw new ArgumentOutOfRangeException("providers", provider,
                        "Unsupported ONNX execution provider.");
                if (!uniqueProviders.Contains(provider))
                    uniqueProviders.Add(provider);
            }
            return uniqueProviders.ToArray();
        }

        private static InferenceSession CreateSession(string modelPath, OnnxExecutionProvider[] providers,
            out OnnxExecutionProvider actualProvider)
        {
            var errors = new List<string>();
            Exception lastException = null;
            foreach (OnnxExecutionProvider provider in providers)
            {
                try
                {
                    InferenceSession inferenceSession = CreateSession(modelPath, provider);
                    actualProvider = provider;
                    return inferenceSession;
                }
                catch (Exception ex)
                {
                    lastException = ex;
                    errors.Add(provider + ": " + ex.Message);
                }
            }

            actualProvider = default(OnnxExecutionProvider);
            throw new InvalidOperationException(
                "Failed to initialize the ONNX model with the requested providers. " +
                string.Join(" | ", errors), lastException);
        }

        private static InferenceSession CreateSession(string modelPath, OnnxExecutionProvider provider)
        {
            if (provider == OnnxExecutionProvider.Cpu)
                return new InferenceSession(modelPath);

            using (var options = new SessionOptions())
            {
                options.AppendExecutionProvider_OpenVINO(
                    provider == OnnxExecutionProvider.OpenVinoGpu ? "GPU" : "CPU");
                return new InferenceSession(modelPath, options);
            }
        }

        private static unsafe void CopyRows(IntPtr source, int sourceRowStride, IntPtr destination,
            int rowBytes, int height)
        {
            if (sourceRowStride == rowBytes)
            {
                long length = (long)rowBytes * height;
                Buffer.MemoryCopy(source.ToPointer(), destination.ToPointer(), length, length);
                return;
            }

            for (int row = 0; row < height; row++)
            {
                IntPtr sourceRow = new IntPtr(source.ToInt64() + (long)row * sourceRowStride);
                IntPtr destinationRow = new IntPtr(destination.ToInt64() + (long)row * rowBytes);
                Buffer.MemoryCopy(sourceRow.ToPointer(), destinationRow.ToPointer(), rowBytes, rowBytes);
            }
        }

        private static IReadOnlyList<OnnxDetection> ApplyClassAwareNms(List<OnnxDetection> candidates,
            float threshold)
        {
            var kept = new List<OnnxDetection>();
            foreach (OnnxDetection candidate in candidates.OrderByDescending(item => item.Confidence))
            {
                if (kept.Any(item => item.ClassIndex == candidate.ClassIndex &&
                    IntersectionOverUnion(item, candidate) > threshold))
                {
                    continue;
                }
                kept.Add(candidate);
            }
            return kept;
        }

        private static float IntersectionOverUnion(OnnxDetection first, OnnxDetection second)
        {
            float intersectionWidth = Math.Max(0,
                Math.Min(first.X2, second.X2) - Math.Max(first.X1, second.X1));
            float intersectionHeight = Math.Max(0,
                Math.Min(first.Y2, second.Y2) - Math.Max(first.Y1, second.Y1));
            float intersection = intersectionWidth * intersectionHeight;
            float union = first.Width * first.Height + second.Width * second.Height - intersection;
            return union <= 0 ? 0 : intersection / union;
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
                {
                    return BuildClassNames((Dictionary<string, string>)serializer.ReadObject(stream));
                }
            }
            catch (SerializationException)
            {
                var matches = Regex.Matches(serialized,
                    "(?<index>\\d+)\\s*:\\s*['\"](?<name>[^'\"]*)['\"]");
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
                if (!int.TryParse(item.Key, out index) || index < 0 || index >= names.Length ||
                    names[index] != null)
                {
                    throw new InvalidOperationException(
                        "Class indices must be unique and contiguous from zero.");
                }
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

        private static float Clamp(float value, float minimum, float maximum)
        {
            return Math.Max(minimum, Math.Min(maximum, value));
        }

        private void ThrowIfDisposed()
        {
            if (disposed)
                throw new ObjectDisposedException("OnnxObjectDetectionModel");
        }

        private sealed class InputWorkspace : IDisposable
        {
            private IntPtr stagingBuffer;
            private int stagingCapacity;
            private long[] stagingShape;
            private OrtValue directInput;
            private IntPtr directPointer;
            private int directLength;
            private long[] directShape;

            public InputWorkspace()
            {
                RunOptions = new RunOptions();
                InputValues = new OrtValue[1];
            }

            public RunOptions RunOptions { get; private set; }
            public OrtValue[] InputValues { get; private set; }
            public OrtValue StagingInput { get; private set; }

            public IntPtr PrepareStagingInput(int length, long[] shape)
            {
                if (stagingCapacity < length)
                {
                    if (StagingInput != null)
                        StagingInput.Dispose();
                    StagingInput = null;
                    if (stagingBuffer != IntPtr.Zero)
                    {
                        Marshal.FreeHGlobal(stagingBuffer);
                        stagingBuffer = IntPtr.Zero;
                    }
                    stagingCapacity = 0;
                    stagingBuffer = Marshal.AllocHGlobal(length);
                    stagingCapacity = length;
                    stagingShape = null;
                }

                if (StagingInput == null || !ShapesEqual(stagingShape, shape))
                {
                    if (StagingInput != null)
                        StagingInput.Dispose();
                    StagingInput = OrtValue.CreateTensorValueWithData(
                        OrtMemoryInfo.DefaultInstance, TensorElementType.UInt8,
                        shape, stagingBuffer, length);
                    stagingShape = (long[])shape.Clone();
                }

                return stagingBuffer;
            }

            public OrtValue PrepareDirectInput(IntPtr pointer, int length, long[] shape)
            {
                if (directInput == null || directPointer != pointer || directLength != length ||
                    !ShapesEqual(directShape, shape))
                {
                    if (directInput != null)
                        directInput.Dispose();
                    directInput = OrtValue.CreateTensorValueWithData(
                        OrtMemoryInfo.DefaultInstance, TensorElementType.UInt8,
                        shape, pointer, length);
                    directPointer = pointer;
                    directLength = length;
                    directShape = (long[])shape.Clone();
                }
                return directInput;
            }

            public void Dispose()
            {
                if (directInput != null)
                    directInput.Dispose();
                if (StagingInput != null)
                    StagingInput.Dispose();
                if (RunOptions != null)
                    RunOptions.Dispose();
                if (stagingBuffer != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(stagingBuffer);
                    stagingBuffer = IntPtr.Zero;
                }
            }

            private static bool ShapesEqual(long[] first, long[] second)
            {
                if (first == null || second == null || first.Length != second.Length)
                    return false;
                for (int i = 0; i < first.Length; i++)
                {
                    if (first[i] != second[i])
                        return false;
                }
                return true;
            }
        }
    }
}
