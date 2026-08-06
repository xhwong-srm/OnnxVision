using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxVision.Imaging;
using OnnxVision.Runtime;

namespace OnnxVision.Detection
{
    /// <summary>
    /// Model-neutral object detector for the onnx-vision-object-detection 2.0.0 contract.
    /// A model instance can be reused and supports concurrent predictions.
    /// </summary>
    public sealed class OnnxObjectDetectionModel : IDisposable
    {
        private const string BoxesOutputName = "boxes";
        private const string ScoresOutputName = "scores";
        private const string ClassIdsOutputName = "class_ids";
        private readonly InferenceSession session;
        private readonly string[] classNames;
        private readonly OnnxPixelFormat requiredPixelFormat;
        private readonly int inputWidth;
        private readonly int inputHeight;
        private readonly int inputBatchDimension;
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
                OnnxContractMetadata contract = OnnxContractMetadata.Read(
                    session, OnnxVisionContract.ObjectDetectionTask,
                    OnnxVisionContract.ObjectDetectionName);
                nmsRequired = contract.NmsRequired;

                var explicitClassNames = classNames == null ? null : classNames.ToArray();
                var embeddedClassNames = contract.ClassNames;
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
                        "ONNX object detection requires embedded preprocessing with uint8 [B,1,H,W] BW8 NCHW or uint8 [B,H,W,3] C24 NHWC BGR input.");
                }
                inputBatchDimension = dimensions[0];
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
        public bool SupportsDynamicBatch { get { return inputBatchDimension <= 0; } }
        public int? FixedBatchSize { get { return inputBatchDimension > 0 ? (int?)inputBatchDimension : null; } }
        public bool RequiresColorInput { get { return requiredPixelFormat == OnnxPixelFormat.Bgr24; } }
        public bool NmsRequired { get { return nmsRequired; } }

        public string InputDescription
        {
            get
            {
                return RequiresColorInput
                    ? "uint8 [B,H,W,3] C24 NHWC BGR; preprocessing runs inside ONNX"
                    : "uint8 [B,1,H,W] BW8 NCHW; preprocessing runs inside ONNX";
            }
        }

        public IReadOnlyList<OnnxDetection> Detect(OnnxImageBuffer image,
            float confidenceThreshold = 0.5f, float nmsIouThreshold = 0.7f)
        {
            ThrowIfDisposed();
            if (inputBatchDimension > 1)
                throw new InvalidOperationException(string.Format(
                    "This model requires batch size {0}; use DetectBatch with exactly that many images.",
                    inputBatchDimension));
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
                return Execute(workspace, inputValue, new[] { image }, confidenceThreshold, nmsIouThreshold)[0];
            }
            finally
            {
                if (disposed)
                    workspace.Dispose();
                else
                    workspaces.Add(workspace);
            }
        }

        public IReadOnlyList<IReadOnlyList<OnnxDetection>> DetectBatch(
            IReadOnlyList<OnnxImageBuffer> images,
            float confidenceThreshold = 0.5f, float nmsIouThreshold = 0.7f)
        {
            ThrowIfDisposed();
            ValidateThresholds(confidenceThreshold, nmsIouThreshold);
            ValidateBatch(images);

            InputWorkspace workspace;
            if (!workspaces.TryTake(out workspace))
                workspace = new InputWorkspace();

            try
            {
                OnnxImageBuffer first = images[0];
                int channels = first.BytesPerPixel;
                int rowBytes = checked(first.Width * channels);
                int imageLength = checked(rowBytes * first.Height);
                int totalLength = checked(imageLength * images.Count);
                long[] inputShape = requiredPixelFormat == OnnxPixelFormat.Bw8
                    ? new long[] { images.Count, 1, first.Height, first.Width }
                    : new long[] { images.Count, first.Height, first.Width, 3 };
                IntPtr destination = workspace.PrepareStagingInput(totalLength, inputShape);
                for (int index = 0; index < images.Count; index++)
                {
                    CopyRows(images[index].Data, images[index].RowStride,
                        new IntPtr(destination.ToInt64() + (long)index * imageLength),
                        rowBytes, first.Height);
                }
                workspace.InputValues[0] = workspace.StagingInput;
                return Execute(workspace, workspace.StagingInput, images, confidenceThreshold, nmsIouThreshold);
            }
            finally
            {
                if (disposed)
                    workspace.Dispose();
                else
                    workspaces.Add(workspace);
            }
        }

        private IReadOnlyList<IReadOnlyList<OnnxDetection>> Execute(
            InputWorkspace workspace, OrtValue inputValue, IReadOnlyList<OnnxImageBuffer> images,
            float confidenceThreshold, float nmsIouThreshold)
        {
            workspace.InputValues[0] = inputValue;
            using (var results = session.Run(workspace.RunOptions, inputNames,
                workspace.InputValues, outputNames))
            {
                var outputs = results.ToArray();
                var boxes = outputs[0].GetTensorDataAsSpan<float>();
                var scores = outputs[1].GetTensorDataAsSpan<float>();
                var classIds = outputs[2].GetTensorDataAsSpan<long>();
                int batchSize = images.Count;
                if (scores.Length % batchSize != 0 || boxes.Length != scores.Length * 4 || classIds.Length != scores.Length)
                    throw new InvalidOperationException("Detection output tensor sizes do not agree with the input batch.");
                int queryCount = scores.Length / batchSize;
                var detections = new List<IReadOnlyList<OnnxDetection>>(batchSize);
                for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
                {
                    OnnxImageBuffer image = images[batchIndex];
                    var candidates = new List<OnnxDetection>();
                    for (int queryIndex = 0; queryIndex < queryCount; queryIndex++)
                    {
                        int scoreIndex = batchIndex * queryCount + queryIndex;
                        float score = scores[scoreIndex];
                        if (float.IsNaN(score) || float.IsInfinity(score) || score < 0 || score > 1)
                            throw new InvalidOperationException("Model returned a non-finite or out-of-range detection score.");
                        if (score <= 0)
                            continue;
                        if (score < confidenceThreshold)
                            continue;

                        long classId = classIds[scoreIndex];
                        if (classId < 0 || classId >= classNames.Length)
                            throw new InvalidOperationException("Model returned an out-of-range class ID.");
                        int classIndex = (int)classId;
                        int boxIndex = scoreIndex * 4;
                        float x1 = boxes[boxIndex];
                        float y1 = boxes[boxIndex + 1];
                        float x2 = boxes[boxIndex + 2];
                        float y2 = boxes[boxIndex + 3];
                        if (!IsNormalizedCoordinate(x1) || !IsNormalizedCoordinate(y1) ||
                            !IsNormalizedCoordinate(x2) || !IsNormalizedCoordinate(y2) ||
                            x2 < x1 || y2 < y1)
                        {
                            throw new InvalidOperationException(
                                "Model returned an invalid normalized xyxy detection box.");
                        }
                        candidates.Add(new OnnxDetection(
                            classNames[classIndex], classIndex, score,
                            x1 * image.Width, y1 * image.Height,
                            x2 * image.Width, y2 * image.Height));
                    }
                    detections.Add(!nmsRequired
                        ? candidates
                        : ApplyClassAwareNms(candidates, nmsIouThreshold));
                }
                return detections;
            }
        }

        private void ValidateBatch(IReadOnlyList<OnnxImageBuffer> images)
        {
            if (images == null || images.Count == 0)
                throw new ArgumentException("At least one image is required.", "images");
            if (inputBatchDimension > 0 && images.Count != inputBatchDimension)
                throw new ArgumentException(string.Format(
                    "The model requires batch size {0}, but {1} images were supplied.",
                    inputBatchDimension, images.Count), "images");

            OnnxImageBuffer first = images[0];
            ValidateImage(first);
            for (int index = 1; index < images.Count; index++)
            {
                ValidateImage(images[index]);
                if (images[index].Width != first.Width || images[index].Height != first.Height)
                    throw new ArgumentException("All images in a batch must have the same dimensions.", "images");
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

        private static bool IsNormalizedCoordinate(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value) && value >= 0 && value <= 1;
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
