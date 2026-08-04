using System;

namespace OnnxVision.Detection
{
    public sealed class OnnxDetection
    {
        internal OnnxDetection(string className, int classIndex, float confidence,
            float x1, float y1, float x2, float y2)
        {
            ClassName = className;
            ClassIndex = classIndex;
            Confidence = confidence;
            X1 = x1;
            Y1 = y1;
            X2 = x2;
            Y2 = y2;
        }

        public string ClassName { get; private set; }
        public string Name { get { return ClassName; } }
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
