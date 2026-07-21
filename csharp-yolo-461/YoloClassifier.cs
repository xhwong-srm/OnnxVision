using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using Euresys.Open_eVision_22_12;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CSharpYolo461
{
    /// <summary>
    /// Reusable ONNX image-classification backend. Keep one instance alive and reuse it.
    /// An instance is not safe for concurrent calls to Predict.
    /// </summary>
    public sealed class YoloClassifier : IDisposable
    {
        private const int FloatModelImageSize = 224;
        private const int FloatModelPlaneSize = FloatModelImageSize * FloatModelImageSize;

        private readonly InferenceSession session;
        private readonly string[] classNames;
        private readonly EROIBW8 attachedBw8Roi;
        private readonly EROIC24 attachedC24Roi;
        private readonly RoiPlacement defaultRoi;
        private readonly InputContract inputContract;
        private readonly float[] floatTensorBuffer;
        private byte[] byteTensorBuffer;
        private int byteTensorWidth;
        private int byteTensorHeight;
        private IReadOnlyCollection<NamedOnnxValue> inputs;
        private long preprocessTicks;
        private long inferenceTicks;
        private bool disposed;

        public YoloClassifier(string modelPath, IEnumerable<string> classNames, RoiPlacement defaultRoi = null,
            ExecutionProvider executionProvider = ExecutionProvider.Cpu)
        {
            if (string.IsNullOrWhiteSpace(modelPath))
                throw new ArgumentException("A model path is required.", "modelPath");
            if (classNames == null)
                throw new ArgumentNullException("classNames");

            this.classNames = classNames.ToArray();
            if (this.classNames.Length == 0 || this.classNames.Any(string.IsNullOrWhiteSpace))
                throw new ArgumentException("At least one non-empty class name is required.", "classNames");

            ExecutionProvider = executionProvider;
            using (var options = CreateSessionOptions(executionProvider))
                session = new InferenceSession(modelPath, options);

            var inputMetadata = session.InputMetadata.Single();
            InputName = inputMetadata.Key;
            var dimensions = inputMetadata.Value.Dimensions;
            if (inputMetadata.Value.ElementType == typeof(float))
                inputContract = InputContract.FloatNchw;
            else if (inputMetadata.Value.ElementType == typeof(byte) && dimensions.Length == 4 && dimensions[1] == 1)
                inputContract = InputContract.Bw8Nchw;
            else if (inputMetadata.Value.ElementType == typeof(byte) && dimensions.Length == 4 && dimensions[3] == 3)
                inputContract = InputContract.C24Nhwc;
            else
                throw new NotSupportedException("Expected float NCHW, uint8 BW8 NCHW, or uint8 C24 NHWC input metadata.");

            this.defaultRoi = defaultRoi;
            if (inputContract == InputContract.C24Nhwc)
                attachedC24Roi = new EROIC24();
            else
                attachedBw8Roi = new EROIBW8();

            if (inputContract == InputContract.FloatNchw)
            {
                floatTensorBuffer = new float[3 * FloatModelPlaneSize];
                var tensor = new DenseTensor<float>(floatTensorBuffer,
                    new[] { 1, 3, FloatModelImageSize, FloatModelImageSize });
                inputs = new[] { NamedOnnxValue.CreateFromTensor(InputName, tensor) };
            }
        }

        public string InputName { get; private set; }
        public ExecutionProvider ExecutionProvider { get; private set; }

        public string InputDescription
        {
            get
            {
                switch (inputContract)
                {
                    case InputContract.FloatNchw:
                        return "float NCHW; resize and normalization run in C#";
                    case InputContract.Bw8Nchw:
                        return "uint8 [1,1,H,W] BW8 NCHW; preprocessing runs inside ONNX";
                    default:
                        return "uint8 [1,H,W,3] C24 NHWC BGR; channel reorder and preprocessing run inside ONNX";
                }
            }
        }

        public string EuresysInputDescription
        {
            get { return inputContract == InputContract.C24Nhwc ? "EImageC24 / EROIC24" : "EImageBW8 / EROIBW8"; }
        }

        public bool HasEmbeddedPreprocessing { get { return inputContract != InputContract.FloatNchw; } }
        public double PreprocessMilliseconds { get { return preprocessTicks * 1000.0 / Stopwatch.Frequency; } }
        public double InferenceMilliseconds { get { return inferenceTicks * 1000.0 / Stopwatch.Frequency; } }

        private static SessionOptions CreateSessionOptions(ExecutionProvider executionProvider)
        {
            var options = new SessionOptions();
            if (executionProvider == ExecutionProvider.DirectML)
            {
                options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                options.EnableMemoryPattern = false;
                options.AppendExecutionProvider_DML(0);
            }
            else if (executionProvider == ExecutionProvider.OpenVinoCpu)
            {
                options.AppendExecutionProvider_OpenVINO("CPU");
            }
            else if (executionProvider == ExecutionProvider.OpenVinoGpu)
            {
                options.AppendExecutionProvider_OpenVINO("GPU");
            }

            return options;
        }

        public Prediction Predict(string imagePath)
        {
            ThrowIfDisposed();
            if (string.IsNullOrWhiteSpace(imagePath))
                throw new ArgumentException("An image path is required.", "imagePath");

            if (inputContract == InputContract.C24Nhwc)
            {
                using (var image = new EImageC24())
                {
                    image.Load(imagePath);
                    return Predict(image);
                }
            }

            using (var image = new EImageBW8())
            {
                image.Load(imagePath);
                return Predict(image);
            }
        }

        public Prediction Predict(EImageBW8 image)
        {
            return Predict(image, defaultRoi);
        }

        public Prediction Predict(EImageBW8 image, RoiPlacement roi)
        {
            ThrowIfDisposed();
            if (image == null)
                throw new ArgumentNullException("image");
            EnsureBw8Model();

            var placement = roi ?? new RoiPlacement(0, 0, image.Width, image.Height);
            ValidatePlacement(placement, image.Width, image.Height);
            attachedBw8Roi.Detach();
            attachedBw8Roi.Attach(image);
            attachedBw8Roi.SetPlacement(placement.X, placement.Y, placement.Width, placement.Height);
            try
            {
                return Predict(attachedBw8Roi, Stopwatch.GetTimestamp());
            }
            finally
            {
                attachedBw8Roi.Detach();
            }
        }

        public Prediction Predict(EImageC24 image)
        {
            return Predict(image, defaultRoi);
        }

        public Prediction Predict(EImageC24 image, RoiPlacement roi)
        {
            ThrowIfDisposed();
            if (image == null)
                throw new ArgumentNullException("image");
            EnsureC24Model();

            var placement = roi ?? new RoiPlacement(0, 0, image.Width, image.Height);
            ValidatePlacement(placement, image.Width, image.Height);
            attachedC24Roi.Detach();
            attachedC24Roi.Attach(image);
            attachedC24Roi.SetPlacement(placement.X, placement.Y, placement.Width, placement.Height);
            try
            {
                return Predict(attachedC24Roi, Stopwatch.GetTimestamp());
            }
            finally
            {
                attachedC24Roi.Detach();
            }
        }

        public Prediction Predict(EROIBW8 roi)
        {
            ThrowIfDisposed();
            if (roi == null)
                throw new ArgumentNullException("roi");
            EnsureBw8Model();
            return Predict(roi, Stopwatch.GetTimestamp());
        }

        public Prediction Predict(EROIC24 roi)
        {
            ThrowIfDisposed();
            if (roi == null)
                throw new ArgumentNullException("roi");
            EnsureC24Model();
            return Predict(roi, Stopwatch.GetTimestamp());
        }

        public void ResetTimings()
        {
            ThrowIfDisposed();
            preprocessTicks = 0;
            inferenceTicks = 0;
        }

        public void Dispose()
        {
            if (disposed)
                return;

            disposed = true;
            if (attachedBw8Roi != null)
                attachedBw8Roi.Dispose();
            if (attachedC24Roi != null)
                attachedC24Roi.Dispose();
            session.Dispose();
        }

        private Prediction Predict(EROIBW8 roi, long started)
        {
            LoadTensor(roi);
            return RunInference(started);
        }

        private Prediction Predict(EROIC24 roi, long started)
        {
            LoadTensor(roi);
            return RunInference(started);
        }

        private Prediction RunInference(long started)
        {
            var preprocessed = Stopwatch.GetTimestamp();
            Prediction prediction;
            using (var results = session.Run(inputs))
            {
                var scores = results.First().AsTensor<float>();
                if (scores.Length != classNames.Length)
                    throw new InvalidOperationException(string.Format(
                        "Model returned {0} scores, but {1} class names were supplied.", scores.Length, classNames.Length));

                var bestIndex = 0;
                var bestScore = scores.GetValue(0);
                for (var index = 1; index < scores.Length; index++)
                {
                    var score = scores.GetValue(index);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestIndex = index;
                    }
                }

                prediction = new Prediction(classNames[bestIndex], bestIndex,
                    SoftmaxConfidence(scores, bestIndex));
            }

            var inferred = Stopwatch.GetTimestamp();
            preprocessTicks += preprocessed - started;
            inferenceTicks += inferred - preprocessed;
            return prediction;
        }

        private unsafe void LoadTensor(EROIBW8 roi)
        {
            if (inputContract == InputContract.Bw8Nchw)
            {
                LoadByteTensor(roi);
                return;
            }

            for (var y = 0; y < FloatModelImageSize; y++)
            {
                var sourceY = ((y + 0.5) * roi.Height / FloatModelImageSize) - 0.5;
                var y0 = Math.Max(0, Math.Min(roi.Height - 1, (int)Math.Floor(sourceY)));
                var y1 = Math.Min(y0 + 1, roi.Height - 1);
                var yWeight = sourceY <= 0 ? 0.0 : sourceY - Math.Floor(sourceY);
                var row0 = (byte*)roi.GetImagePtr(0, y0).ToPointer();
                var row1 = (byte*)roi.GetImagePtr(0, y1).ToPointer();
                var tensorRow = y * FloatModelImageSize;
                for (var x = 0; x < FloatModelImageSize; x++)
                {
                    var sourceX = ((x + 0.5) * roi.Width / FloatModelImageSize) - 0.5;
                    var x0 = Math.Max(0, Math.Min(roi.Width - 1, (int)Math.Floor(sourceX)));
                    var x1 = Math.Min(x0 + 1, roi.Width - 1);
                    var xWeight = sourceX <= 0 ? 0.0 : sourceX - Math.Floor(sourceX);
                    var top = row0[x0] + (row0[x1] - row0[x0]) * xWeight;
                    var bottom = row1[x0] + (row1[x1] - row1[x0]) * xWeight;
                    var pixel = (float)((top + (bottom - top) * yWeight) / 255.0);
                    var tensorOffset = tensorRow + x;
                    floatTensorBuffer[tensorOffset] = pixel;
                    floatTensorBuffer[FloatModelPlaneSize + tensorOffset] = pixel;
                    floatTensorBuffer[2 * FloatModelPlaneSize + tensorOffset] = pixel;
                }
            }
        }

        private void LoadByteTensor(EROIBW8 roi)
        {
            EnsureByteTensor(roi.Width, roi.Height, 1);
            for (var y = 0; y < roi.Height; y++)
                Marshal.Copy(roi.GetImagePtr(0, y), byteTensorBuffer, y * roi.Width, roi.Width);
        }

        private void LoadTensor(EROIC24 roi)
        {
            var rowBytes = checked(roi.Width * 3);
            EnsureByteTensor(roi.Width, roi.Height, 3);
            for (var y = 0; y < roi.Height; y++)
                Marshal.Copy(roi.GetImagePtr(0, y), byteTensorBuffer, y * rowBytes, rowBytes);
        }

        private void EnsureByteTensor(int width, int height, int channels)
        {
            if (width <= 0 || height <= 0)
                throw new ArgumentException("The input ROI must have positive width and height.", "roi");

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
            var tensor = new DenseTensor<byte>(byteTensorBuffer, shape);
            inputs = new[] { NamedOnnxValue.CreateFromTensor(InputName, tensor) };
        }

        private void EnsureBw8Model()
        {
            if (inputContract == InputContract.C24Nhwc)
                throw new InvalidOperationException("This model requires EImageC24 or EROIC24 input.");
        }

        private void EnsureC24Model()
        {
            if (inputContract != InputContract.C24Nhwc)
                throw new InvalidOperationException("This model requires EImageBW8 or EROIBW8 input.");
        }

        private static void ValidatePlacement(RoiPlacement placement, int imageWidth, int imageHeight)
        {
            if (placement.X < 0 || placement.Y < 0 ||
                placement.X + placement.Width > imageWidth ||
                placement.Y + placement.Height > imageHeight)
                throw new ArgumentOutOfRangeException("roi",
                    string.Format("ROI ({0}, {1}, {2}, {3}) is outside image {4}x{5}.",
                        placement.X, placement.Y, placement.Width, placement.Height, imageWidth, imageHeight));
        }

        private static float SoftmaxConfidence(Tensor<float> scores, int selectedIndex)
        {
            var maximum = float.MinValue;
            for (var index = 0; index < scores.Length; index++)
                maximum = Math.Max(maximum, scores.GetValue(index));

            var denominator = 0.0;
            for (var index = 0; index < scores.Length; index++)
                denominator += Math.Exp(scores.GetValue(index) - maximum);

            return (float)(Math.Exp(scores.GetValue(selectedIndex) - maximum) / denominator);
        }

        private void ThrowIfDisposed()
        {
            if (disposed)
                throw new ObjectDisposedException("YoloClassifier");
        }

        private enum InputContract
        {
            FloatNchw,
            Bw8Nchw,
            C24Nhwc
        }
    }

    public enum ExecutionProvider
    {
        Cpu,
        DirectML,
        OpenVinoCpu,
        OpenVinoGpu
    }

    public sealed class RoiPlacement
    {
        public RoiPlacement(int x, int y, int width, int height)
        {
            if (width <= 0)
                throw new ArgumentOutOfRangeException("width", "ROI width must be positive.");
            if (height <= 0)
                throw new ArgumentOutOfRangeException("height", "ROI height must be positive.");

            X = x;
            Y = y;
            Width = width;
            Height = height;
        }

        public int X { get; private set; }
        public int Y { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
    }

    public sealed class Prediction
    {
        public Prediction(string name, int classIndex, float confidence)
        {
            Name = name;
            ClassIndex = classIndex;
            Confidence = confidence;
        }

        public string Name { get; private set; }
        public int ClassIndex { get; private set; }
        public float Confidence { get; private set; }
    }
}
