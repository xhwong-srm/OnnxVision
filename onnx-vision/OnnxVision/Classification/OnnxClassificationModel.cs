using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
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

namespace OnnxVision.Classification
{
    public sealed class OnnxClassificationModel : IDisposable
    {
        private const string ClassNamesMetadataKey = "names";
        private readonly InferenceSession session;
        private readonly string inputName;
        private readonly string outputName;
        private readonly string[] classNames;
        private readonly OnnxPixelFormat requiredPixelFormat;
        private readonly int inputWidth;
        private readonly int inputHeight;
        private readonly int inputBatchDimension;
        private readonly int classCount;
        private readonly string[] inputNames;
        private readonly string[] outputNames;
        private readonly ConcurrentBag<InputWorkspace> workspaces = new ConcurrentBag<InputWorkspace>();
        private bool disposed;

        public OnnxClassificationModel(string modelPath, params OnnxExecutionProvider[] providerPriority)
        {
            if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
                throw new FileNotFoundException("ONNX model file was not found.", modelPath);
            if (providerPriority == null || providerPriority.Length == 0)
                throw new ArgumentException("At least one ONNX execution provider is required.", "providerPriority");

            var providers = ValidateProviders(providerPriority);
            RequestedProviders = providers;
            session = CreateSession(modelPath, providers, out OnnxExecutionProvider actualProvider);
            ActualProvider = actualProvider;

            OnnxContractMetadata contract = OnnxContractMetadata.Read(
                session,
                OnnxVisionContract.ClassificationTask,
                OnnxVisionContract.ClassificationName);

            var input = session.InputMetadata.Single();
            inputName = input.Key;
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
                session.Dispose();
                throw new NotSupportedException(
                    "ONNX classification requires embedded preprocessing with uint8 [B,1,H,W] BW8 NCHW or uint8 [B,H,W,3] C24 NHWC BGR input.");
            }
            inputBatchDimension = dimensions[0];

            try
            {
                classNames = contract.ClassNames;
                outputName = "probabilities";
                classCount = ResolveClassCount(
                    session.OutputMetadata[outputName], classNames.Length, inputBatchDimension);
                inputNames = new[] { inputName };
                outputNames = new[] { outputName };
            }
            catch
            {
                session.Dispose();
                throw;
            }
        }

        public IReadOnlyList<OnnxExecutionProvider> RequestedProviders { get; private set; }
        public OnnxExecutionProvider ActualProvider { get; private set; }
        public IReadOnlyList<string> ClassNames { get { return classNames; } }
        public OnnxPixelFormat RequiredPixelFormat { get { return requiredPixelFormat; } }
        public int InputWidth { get { return inputWidth; } }
        public int InputHeight { get { return inputHeight; } }
        public bool SupportsDynamicBatch { get { return inputBatchDimension <= 0; } }
        public int? FixedBatchSize { get { return SupportsDynamicBatch ? (int?)null : inputBatchDimension; } }
        public bool RequiresColorInput { get { return requiredPixelFormat == OnnxPixelFormat.Bgr24; } }

        public OnnxClassification Classify(OnnxImageBuffer image)
        {
            ThrowIfDisposed();
            if (!SupportsDynamicBatch && inputBatchDimension != 1)
                throw new NotSupportedException(string.Format(
                    "This model requires batch size {0}; use ClassifyBatch with exactly that many images.",
                    inputBatchDimension));
            ValidateImage(image);

            InputWorkspace workspace;
            if (!workspaces.TryTake(out workspace))
                workspace = new InputWorkspace();

            try
            {
                int channels = image.BytesPerPixel;
                int rowBytes = checked(image.Width * channels);
                int requiredLength = checked(rowBytes * image.Height);
                long[] inputShape =
                    requiredPixelFormat == OnnxPixelFormat.Bw8
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
                return Execute(workspace, inputValue, 1)[0];
            }
            finally
            {
                if (disposed)
                    workspace.Dispose();
                else
                    workspaces.Add(workspace);
            }
        }

        public IReadOnlyList<OnnxClassification> ClassifyBatch(IReadOnlyList<OnnxImageBuffer> images)
        {
            ThrowIfDisposed();
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
                return Execute(workspace, workspace.StagingInput, images.Count);
            }
            finally
            {
                if (disposed)
                    workspace.Dispose();
                else
                    workspaces.Add(workspace);
            }
        }

        private IReadOnlyList<OnnxClassification> Execute(
            InputWorkspace workspace, OrtValue inputValue, int batchSize)
        {
            workspace.PrepareOutput(batchSize, classCount);
            workspace.InputValues[0] = inputValue;
            long started = Stopwatch.GetTimestamp();
            session.Run(workspace.RunOptions, inputNames, workspace.InputValues, outputNames, workspace.OutputValues);
            var scores = workspace.Output.GetTensorDataAsSpan<float>();
            int expectedLength = checked(batchSize * classCount);
            if (scores.Length != expectedLength)
            {
                throw new InvalidOperationException(string.Format(
                    "Model returned {0} scores for batch {1} and {2} embedded classes.",
                    scores.Length, batchSize, classCount));
            }

            double inferenceMilliseconds =
                (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
            var results = new List<OnnxClassification>(batchSize);
            for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
            {
                int bestIndex;
                float[] probabilities = ToProbabilities(
                    scores.Slice(batchIndex * classCount, classCount), out bestIndex);
                results.Add(new OnnxClassification(classNames[bestIndex], bestIndex,
                    probabilities[bestIndex], probabilities, inferenceMilliseconds));
            }
            return results;
        }

        private void ValidateBatch(IReadOnlyList<OnnxImageBuffer> images)
        {
            if (images == null || images.Count == 0)
                throw new ArgumentException("At least one image is required.", "images");
            if (!SupportsDynamicBatch && images.Count != inputBatchDimension)
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
                    throw new ArgumentOutOfRangeException("providers", provider, "Unsupported ONNX execution provider.");
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
                "Failed to initialize the ONNX model with the requested providers. " + string.Join(" | ", errors),
                lastException);
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

        private static int ResolveClassCount(NodeMetadata output, int classCount, int batchDimension)
        {
            if (output.ElementType != typeof(float))
                throw new NotSupportedException("ONNX classification output must contain float scores.");

            int[] dimensions = output.Dimensions;
            if (dimensions.Length == 2 &&
                ((batchDimension <= 0 && dimensions[0] <= 0) ||
                 (batchDimension > 0 && dimensions[0] == batchDimension)) &&
                (dimensions[1] <= 0 || dimensions[1] == classCount))
                return classCount;

            throw new NotSupportedException(string.Format(
                "ONNX classification output must have shape [B,{0}] matching the model batch contract.", classCount));
        }

        private static float[] ToProbabilities(ReadOnlySpan<float> scores, out int bestIndex)
        {
            var probabilities = new float[scores.Length];
            double sum = 0;
            bestIndex = 0;
            for (int i = 0; i < scores.Length; i++)
            {
                float value = scores[i];
                probabilities[i] = value;
                if (float.IsNaN(value) || float.IsInfinity(value) || value < 0f || value > 1f)
                    throw new InvalidOperationException(
                        "Classification output must contain finite probabilities within [0,1].");
                if (i > 0 && value > probabilities[bestIndex])
                    bestIndex = i;
                sum += value;
            }
            if (Math.Abs(sum - 1.0) > 0.001)
                throw new InvalidOperationException(
                    "Classification probability rows must sum to one.");
            return probabilities;
        }

        private static string[] ReadEmbeddedClassNames(InferenceSession inferenceSession)
        {
            string serializedNames;
            if (!inferenceSession.ModelMetadata.CustomMetadataMap.TryGetValue(
                    ClassNamesMetadataKey, out serializedNames) ||
                string.IsNullOrWhiteSpace(serializedNames))
            {
                return null;
            }
            serializedNames = serializedNames.Trim().Trim('\'');

            if (serializedNames.StartsWith("{", StringComparison.Ordinal))
            {
                try
                {
                    var serializer = new DataContractJsonSerializer(typeof(Dictionary<string, string>),
                        new DataContractJsonSerializerSettings { UseSimpleDictionaryFormat = true });
                    using (var stream = new MemoryStream(Encoding.UTF8.GetBytes(serializedNames)))
                        return BuildClassNames((Dictionary<string, string>)serializer.ReadObject(stream));
                }
                catch (SerializationException)
                {
                }
            }

            var matches = Regex.Matches(serializedNames, @"(?<index>\d+)\s*:\s*'(?<name>[^']*)'");
            if (matches.Count == 0)
                matches = Regex.Matches(serializedNames, "(?<index>\\d+)\\s*:\\s*\"(?<name>[^\"]*)\"");
            if (matches.Count == 0)
                throw new InvalidOperationException("The ONNX model 'names' metadata is not a valid class mapping.");

            var names = new string[matches.Count];
            for (int i = 0; i < matches.Count; i++)
            {
                int index = int.Parse(matches[i].Groups["index"].Value);
                if (index != i)
                    throw new InvalidOperationException("ONNX class indices must be contiguous and start at zero.");
                names[i] = matches[i].Groups["name"].Value;
            }
            return names;
        }

        private static string[] BuildClassNames(IDictionary<string, string> mapping)
        {
            var names = new string[mapping.Count];
            foreach (var entry in mapping)
            {
                int index;
                if (!int.TryParse(entry.Key, out index) ||
                    index < 0 || index >= names.Length || names[index] != null)
                {
                    throw new InvalidOperationException(
                        "ONNX class indices must be unique, contiguous, and start at zero.");
                }
                names[index] = entry.Value;
            }
            if (names.Any(string.IsNullOrWhiteSpace))
                throw new InvalidOperationException("ONNX class names cannot be empty.");
            return names;
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

        private void ThrowIfDisposed()
        {
            if (disposed)
                throw new ObjectDisposedException("OnnxClassificationModel");
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
            private long[] outputShape;

            public InputWorkspace()
            {
                RunOptions = new RunOptions();
                InputValues = new OrtValue[1];
            }

            public RunOptions RunOptions { get; private set; }
            public OrtValue[] InputValues { get; private set; }
            public OrtValue[] OutputValues { get; private set; }
            public OrtValue StagingInput { get; private set; }
            public OrtValue Output { get; private set; }

            public void PrepareOutput(int batchSize, int classCount)
            {
                long[] shape = new long[] { batchSize, classCount };
                if (Output != null && ShapesEqual(outputShape, shape))
                    return;
                if (Output != null)
                    Output.Dispose();
                Output = OrtValue.CreateAllocatedTensorValue(
                    OrtAllocator.DefaultInstance, TensorElementType.Float, shape);
                OutputValues = new[] { Output };
                outputShape = shape;
            }

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
                        OrtMemoryInfo.DefaultInstance, TensorElementType.UInt8, shape, stagingBuffer, length);
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
                        OrtMemoryInfo.DefaultInstance, TensorElementType.UInt8, shape, pointer, length);
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
                if (Output != null)
                    Output.Dispose();
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
