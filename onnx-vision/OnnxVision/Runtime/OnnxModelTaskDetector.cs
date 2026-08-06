using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime;

namespace OnnxVision.Runtime
{
    public enum OnnxVisionTask
    {
        Classification,
        ObjectDetection
    }

    /// <summary>
    /// Identifies the ONNX task from the required embedded metadata contract.
    /// Tensor validation is performed by the task-specific model implementation.
    /// </summary>
    public static class OnnxModelTaskDetector
    {
        public static OnnxVisionTask Detect(string modelPath)
        {
            if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
                throw new FileNotFoundException("ONNX model file was not found.", modelPath);

            using (var session = new InferenceSession(modelPath))
            {
                IReadOnlyDictionary<string, string> metadata = session.ModelMetadata.CustomMetadataMap;
                string task;
                if (!metadata.TryGetValue("vision_task", out task))
                {
                    throw new NotSupportedException(
                        "The ONNX model is missing required vision_task metadata.");
                }

                if (string.Equals(task, OnnxVisionContract.ObjectDetectionTask,
                    StringComparison.Ordinal))
                {
                    ValidateContractMetadata(metadata, OnnxVisionContract.ObjectDetectionName);
                    ValidateNamesMetadata(metadata);
                    ValidateNmsMetadata(metadata);
                    return OnnxVisionTask.ObjectDetection;
                }

                if (string.Equals(task, OnnxVisionContract.ClassificationTask,
                    StringComparison.Ordinal))
                {
                    ValidateContractMetadata(metadata, OnnxVisionContract.ClassificationName);
                    ValidateNamesMetadata(metadata);
                    return OnnxVisionTask.Classification;
                }

                throw new NotSupportedException(
                    "Unsupported ONNX vision_task metadata: " + task);
            }
        }

        private static void ValidateContractMetadata(
            IReadOnlyDictionary<string, string> metadata, string expectedName)
        {
            string actualName;
            if (!metadata.TryGetValue("contract_name", out actualName) ||
                !string.Equals(actualName, expectedName, StringComparison.Ordinal))
            {
                throw new NotSupportedException(
                    "Expected contract_name metadata '" + expectedName + "'.");
            }

            string inputs;
            string outputs;
            if (!metadata.TryGetValue("inputs", out inputs) || string.IsNullOrWhiteSpace(inputs) ||
                !metadata.TryGetValue("outputs", out outputs) || string.IsNullOrWhiteSpace(outputs))
            {
                throw new NotSupportedException(
                    "The ONNX model is missing inputs or outputs contract metadata.");
            }

            string version;
            if (!metadata.TryGetValue("contract_version", out version) ||
                !string.Equals(version, OnnxVisionContract.Version, StringComparison.Ordinal))
            {
                throw new NotSupportedException(
                    "Expected contract_version metadata '" + OnnxVisionContract.Version + "'.");
            }
        }

        private static void ValidateNamesMetadata(IReadOnlyDictionary<string, string> metadata)
        {
            string names;
            if (!metadata.TryGetValue("names", out names) || string.IsNullOrWhiteSpace(names))
            {
                throw new NotSupportedException(
                    "The ONNX model must contain names metadata.");
            }
        }

        private static void ValidateNmsMetadata(IReadOnlyDictionary<string, string> metadata)
        {
            string nmsRequired;
            bool parsedNmsRequired;
            if (!metadata.TryGetValue("nms_required", out nmsRequired) ||
                !bool.TryParse(nmsRequired, out parsedNmsRequired))
            {
                throw new NotSupportedException(
                    "Expected boolean nms_required metadata in the object-detection model.");
            }
        }
    }
}
