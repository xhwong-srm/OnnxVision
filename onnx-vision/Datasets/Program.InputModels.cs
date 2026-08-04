using System.Collections.Generic;
using System.Runtime.Serialization;

namespace OnnxVision
{
    internal static partial class Program
    {
        private sealed class ClassificationSample
        {
            public ClassificationSample(string path, string expectedClassName)
            {
                Path = path;
                ExpectedClassName = expectedClassName;
            }

            public string Path { get; private set; }
            public string ExpectedClassName { get; private set; }
        }

        private sealed class ClassificationInput
        {
            public ClassificationInput(List<ClassificationSample> samples, bool isDataset,
                string datasetFormat, string datasetSplit)
            {
                Samples = samples;
                IsDataset = isDataset;
                DatasetFormat = datasetFormat;
                DatasetSplit = datasetSplit;
            }

            public List<ClassificationSample> Samples { get; private set; }
            public bool IsDataset { get; private set; }
            public string DatasetFormat { get; private set; }
            public string DatasetSplit { get; private set; }
        }

        private sealed class DetectionSample
        {
            public DetectionSample(string path, List<GroundTruthDetection> groundTruths)
            {
                Path = path;
                GroundTruths = groundTruths;
            }

            public string Path { get; private set; }
            public List<GroundTruthDetection> GroundTruths { get; private set; }
        }

        private sealed class DetectionInput
        {
            public DetectionInput(List<DetectionSample> samples, bool isDataset,
                string datasetFormat, string datasetSplit, string[] datasetClassNames)
            {
                Samples = samples;
                IsDataset = isDataset;
                DatasetFormat = datasetFormat;
                DatasetSplit = datasetSplit;
                DatasetClassNames = datasetClassNames;
            }

            public List<DetectionSample> Samples { get; private set; }
            public bool IsDataset { get; private set; }
            public string DatasetFormat { get; private set; }
            public string DatasetSplit { get; private set; }
            public string[] DatasetClassNames { get; private set; }
        }

        private sealed class GroundTruthDetection
        {
            public GroundTruthDetection(string className, float x1, float y1, float x2, float y2)
            {
                ClassName = className;
                X1 = x1;
                Y1 = y1;
                X2 = x2;
                Y2 = y2;
            }

            public string ClassName { get; private set; }
            public float X1 { get; private set; }
            public float Y1 { get; private set; }
            public float X2 { get; private set; }
            public float Y2 { get; private set; }
        }

        [DataContract]
        private sealed class CocoDocument
        {
            [DataMember(Name = "images")]
            public List<CocoImage> Images { get; set; }

            [DataMember(Name = "annotations")]
            public List<CocoAnnotation> Annotations { get; set; }

            [DataMember(Name = "categories")]
            public List<CocoCategory> Categories { get; set; }
        }

        [DataContract]
        private sealed class CocoImage
        {
            [DataMember(Name = "id")]
            public long Id { get; set; }

            [DataMember(Name = "file_name")]
            public string FileName { get; set; }
        }

        [DataContract]
        private sealed class CocoAnnotation
        {
            [DataMember(Name = "image_id")]
            public long ImageId { get; set; }

            [DataMember(Name = "category_id")]
            public long CategoryId { get; set; }

            [DataMember(Name = "bbox")]
            public double[] BoundingBox { get; set; }
        }

        [DataContract]
        private sealed class CocoCategory
        {
            [DataMember(Name = "id")]
            public long Id { get; set; }

            [DataMember(Name = "name")]
            public string Name { get; set; }
        }
    }
}
