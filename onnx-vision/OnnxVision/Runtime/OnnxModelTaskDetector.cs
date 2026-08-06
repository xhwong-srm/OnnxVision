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
                    OnnxContractMetadata.Read(session,
                        OnnxVisionContract.ObjectDetectionTask,
                        OnnxVisionContract.ObjectDetectionName);
                    return OnnxVisionTask.ObjectDetection;
                }

                if (string.Equals(task, OnnxVisionContract.ClassificationTask,
                    StringComparison.Ordinal))
                {
                    OnnxContractMetadata.Read(session,
                        OnnxVisionContract.ClassificationTask,
                        OnnxVisionContract.ClassificationName);
                    return OnnxVisionTask.Classification;
                }

                throw new NotSupportedException(
                    "Unsupported ONNX vision_task metadata: " + task);
            }
        }

    }
}
