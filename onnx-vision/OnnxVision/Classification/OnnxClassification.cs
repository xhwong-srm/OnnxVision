using System.Collections.Generic;

namespace OnnxVision.Classification
{
    public sealed class OnnxClassification
    {
        internal OnnxClassification(string className, int classIndex, float confidence,
            IReadOnlyList<float> probabilities, double inferenceMilliseconds)
        {
            ClassName = className;
            ClassIndex = classIndex;
            Confidence = confidence;
            Probabilities = probabilities;
            InferenceMilliseconds = inferenceMilliseconds;
        }

        public string ClassName { get; private set; }
        public int ClassIndex { get; private set; }
        public float Confidence { get; private set; }
        public IReadOnlyList<float> Probabilities { get; private set; }
        public double InferenceMilliseconds { get; private set; }
    }
}
