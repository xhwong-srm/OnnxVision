using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;

namespace OnnxVision
{
    internal static partial class Program
    {
        private static string[] EnumerateImages(string directory)
        {
            return Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories)
                .Where(path => Extensions.Contains(Path.GetExtension(path)))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }

        private static ClassificationInput LoadClassificationInput(string source,
            InputOptions options)
        {
            if (File.Exists(source))
            {
                if (options.ForceDataset || options.Validate || options.Set != null)
                    throw new InvalidOperationException(
                        "A classification dataset must be a directory; a single image cannot be validated.");
                if (!Extensions.Contains(Path.GetExtension(source)))
                    throw new InvalidOperationException("The input file is not a supported image.");
                return new ClassificationInput(
                    new List<ClassificationSample> { new ClassificationSample(source, null) },
                    false, null, null);
            }

            if (!Directory.Exists(source))
                throw new DirectoryNotFoundException("Image or classification dataset does not exist: " + source);

            bool isDataset = options.ForceDataset || IsClassificationDatasetRoot(source);
            if (!isDataset)
            {
                return new ClassificationInput(
                    EnumerateImages(source).Select(path => new ClassificationSample(path, null)).ToList(),
                    false, null, null);
            }

            string split;
            string splitDirectory = ResolveClassificationSplitDirectory(source, options.Set, out split);
            List<ClassificationSample> samples = LoadClassificationSamples(splitDirectory);
            return new ClassificationInput(samples, true, "imagenet", split);
        }

        private static bool IsClassificationDatasetRoot(string root)
        {
            if (FindCocoAnnotation(root, null) != null)
                return false;
            return HasClassDirectoriesWithImages(root) ||
                new[] { "train", "val", "valid", "test" }
                    .Select(split => Path.Combine(root, split))
                    .Any(path => Directory.Exists(path) && HasClassDirectoriesWithImages(path));
        }

        private static bool HasClassDirectoriesWithImages(string root)
        {
            if (!Directory.Exists(root))
                return false;
            foreach (string directory in Directory.EnumerateDirectories(root))
            {
                if (Directory.EnumerateFiles(directory, "*", SearchOption.TopDirectoryOnly)
                    .Any(path => Extensions.Contains(Path.GetExtension(path))))
                {
                    return true;
                }
            }
            return false;
        }

        private static string ResolveClassificationSplitDirectory(string root,
            string requestedSplit, out string selectedSplit)
        {
            selectedSplit = requestedSplit;
            if (requestedSplit == null)
            {
                foreach (string candidate in new[] { "val", "valid", "train", "test" })
                {
                    string candidatePath = Path.Combine(root, candidate);
                    if (Directory.Exists(candidatePath) && HasClassDirectoriesWithImages(candidatePath))
                    {
                        selectedSplit = candidate == "valid" ? "val" : candidate;
                        return candidatePath;
                    }
                }

                if (HasClassDirectoriesWithImages(root))
                {
                    selectedSplit = "root";
                    return root;
                }

                throw new InvalidOperationException(
                    "The classification dataset does not contain train, val, or test class folders.");
            }

            string splitDirectory = Path.Combine(root, requestedSplit);
            if (requestedSplit == "val" && !Directory.Exists(splitDirectory))
                splitDirectory = Path.Combine(root, "valid");
            if (!Directory.Exists(splitDirectory) || !HasClassDirectoriesWithImages(splitDirectory))
            {
                throw new InvalidOperationException(string.Format(CultureInfo.InvariantCulture,
                    "The classification dataset does not contain a labeled '{0}' split.",
                    requestedSplit));
            }
            return splitDirectory;
        }

        private static List<ClassificationSample> LoadClassificationSamples(string root)
        {
            var samples = new List<ClassificationSample>();
            foreach (string classDirectory in Directory.EnumerateDirectories(root)
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
            {
                string className = new DirectoryInfo(classDirectory).Name;
                foreach (string imagePath in EnumerateImages(classDirectory))
                    samples.Add(new ClassificationSample(imagePath, className));
            }
            return samples;
        }

        private static DetectionInput LoadDetectionInput(string source, InputOptions options)
        {
            if (File.Exists(source) && !IsJsonFile(source))
            {
                if (options.ForceDataset || options.Validate || options.Set != null)
                    throw new InvalidOperationException(
                        "A COCO dataset is required for -validate and -set; a single image has no labels.");
                if (!Extensions.Contains(Path.GetExtension(source)))
                    throw new InvalidOperationException("The input file is not a supported image.");
                return new DetectionInput(
                    new List<DetectionSample> { new DetectionSample(source, new List<GroundTruthDetection>()) },
                    false, null, null, null);
            }

            bool datasetRequested = options.ForceDataset || IsCocoDatasetSource(source);
            if (datasetRequested)
            {
                string root = Directory.Exists(source) ? source : InferCocoRoot(source);
                string annotationPath = FindCocoAnnotation(source, options.Set);
                if (annotationPath == null)
                    throw new InvalidOperationException(
                        "The COCO dataset does not contain annotations for the requested split.");

                string split = options.Set ?? InferCocoSplit(annotationPath);
                return LoadCocoInput(root, annotationPath, split);
            }

            if (!Directory.Exists(source))
                throw new DirectoryNotFoundException("Image or COCO dataset does not exist: " + source);
            if (options.Validate || options.Set != null)
                throw new InvalidOperationException("-validate and -set require a COCO detection dataset.");

            return new DetectionInput(
                EnumerateImages(source)
                    .Select(path => new DetectionSample(path, new List<GroundTruthDetection>()))
                    .ToList(), false, null, null, null);
        }

        private static bool IsCocoDatasetSource(string source)
        {
            if (File.Exists(source))
                return IsJsonFile(source);
            return Directory.Exists(source) && FindCocoAnnotation(source, null) != null;
        }

        private static bool IsJsonFile(string path)
        {
            return string.Equals(Path.GetExtension(path), ".json",
                StringComparison.OrdinalIgnoreCase);
        }

        private static string FindCocoAnnotation(string source, string requestedSplit)
        {
            if (File.Exists(source))
                return IsJsonFile(source) ? source : null;
            if (!Directory.Exists(source))
                return null;

            string[] splits = requestedSplit == null
                ? new[] { "val", "train", "test" }
                : new[] { requestedSplit };
            foreach (string split in splits)
            {
                foreach (string candidate in CocoAnnotationCandidates(source, split))
                {
                    if (File.Exists(candidate))
                        return candidate;
                }
            }
            return null;
        }

        private static IEnumerable<string> CocoAnnotationCandidates(string root, string split)
        {
            string canonical = split == "valid" ? "val" : split;
            string[] splitNames = canonical == "val"
                ? new[] { "val", "valid" }
                : new[] { canonical };
            foreach (string name in splitNames)
            {
                yield return Path.Combine(root, "annotations", "instances_" + name + ".json");
                yield return Path.Combine(root, "annotations", "instances_" + name + "2017.json");
                yield return Path.Combine(root, "annotations", "instances_" + name + "_2017.json");
                yield return Path.Combine(root, name, "_annotations.coco.json");
                yield return Path.Combine(root, name, "annotations.json");
                yield return Path.Combine(root, "instances_" + name + ".json");
                yield return Path.Combine(root, name + ".json");
            }

            string splitDirectory = Path.Combine(root, canonical);
            if (canonical == "val" && !Directory.Exists(splitDirectory))
                splitDirectory = Path.Combine(root, "valid");
            if (File.Exists(Path.Combine(splitDirectory, "_annotations.coco.json")))
                yield return Path.Combine(splitDirectory, "_annotations.coco.json");

            string annotationDirectory = Path.Combine(root, "annotations");
            if (Directory.Exists(annotationDirectory))
            {
                foreach (string path in Directory.EnumerateFiles(annotationDirectory, "*.json",
                    SearchOption.TopDirectoryOnly).OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
                {
                    string name = Path.GetFileName(path).ToLowerInvariant();
                    if (name.Contains(canonical) || (canonical == "val" && name.Contains("valid")))
                        yield return path;
                }
            }
        }

        private static string InferCocoRoot(string annotationPath)
        {
            string parent = Path.GetDirectoryName(annotationPath);
            if (parent != null && string.Equals(new DirectoryInfo(parent).Name, "annotations",
                StringComparison.OrdinalIgnoreCase))
            {
                return Directory.GetParent(parent).FullName;
            }
            return parent;
        }

        private static string InferCocoSplit(string annotationPath)
        {
            string name = Path.GetFileName(annotationPath).ToLowerInvariant();
            if (name.Contains("val") || name.Contains("valid"))
                return "val";
            if (name.Contains("test"))
                return "test";
            string parent = Path.GetDirectoryName(annotationPath);
            if (parent != null)
            {
                string parentName = new DirectoryInfo(parent).Name.ToLowerInvariant();
                if (parentName == "val" || parentName == "valid")
                    return "val";
                if (parentName == "test")
                    return "test";
            }
            return "train";
        }

        private static DetectionInput LoadCocoInput(string root, string annotationPath,
            string split)
        {
            CocoDocument document;
            var serializer = new DataContractJsonSerializer(typeof(CocoDocument));
            using (var stream = File.OpenRead(annotationPath))
                document = (CocoDocument)serializer.ReadObject(stream);
            if (document == null || document.Images == null || document.Annotations == null ||
                document.Categories == null)
            {
                throw new InvalidOperationException(
                    "COCO annotations must contain images, annotations, and categories arrays.");
            }

            var categoryNames = new Dictionary<long, string>();
            foreach (CocoCategory category in document.Categories)
            {
                if (category == null || string.IsNullOrWhiteSpace(category.Name))
                    throw new InvalidOperationException("COCO contains an empty category name.");
                if (categoryNames.ContainsKey(category.Id))
                    throw new InvalidOperationException("COCO contains duplicate category IDs.");
                categoryNames.Add(category.Id, category.Name.Trim());
            }

            var annotationsByImage = new Dictionary<long, List<GroundTruthDetection>>();
            foreach (CocoAnnotation annotation in document.Annotations)
            {
                string className;
                if (!categoryNames.TryGetValue(annotation.CategoryId, out className))
                    throw new InvalidOperationException("COCO annotation references an unknown category ID.");
                if (annotation.BoundingBox == null || annotation.BoundingBox.Length < 4)
                    throw new InvalidOperationException("COCO contains an annotation without a valid bbox.");
                double x = annotation.BoundingBox[0];
                double y = annotation.BoundingBox[1];
                double width = annotation.BoundingBox[2];
                double height = annotation.BoundingBox[3];
                if (width <= 0 || height <= 0)
                    continue;
                List<GroundTruthDetection> imageAnnotations;
                if (!annotationsByImage.TryGetValue(annotation.ImageId, out imageAnnotations))
                {
                    imageAnnotations = new List<GroundTruthDetection>();
                    annotationsByImage.Add(annotation.ImageId, imageAnnotations);
                }
                imageAnnotations.Add(new GroundTruthDetection(className,
                    (float)x, (float)y, (float)(x + width), (float)(y + height)));
            }

            var samples = new List<DetectionSample>();
            foreach (CocoImage image in document.Images)
            {
                string imagePath = ResolveCocoImagePath(root, split, image.FileName);
                List<GroundTruthDetection> groundTruths;
                if (!annotationsByImage.TryGetValue(image.Id, out groundTruths))
                    groundTruths = new List<GroundTruthDetection>();
                samples.Add(new DetectionSample(imagePath, groundTruths));
            }
            return new DetectionInput(samples, true, "coco", split,
                categoryNames.Values.Distinct(StringComparer.OrdinalIgnoreCase)
                    .OrderBy(value => value, StringComparer.OrdinalIgnoreCase).ToArray());
        }

        private static string ResolveCocoImagePath(string root, string split, string fileName)
        {
            if (string.IsNullOrWhiteSpace(fileName))
                throw new InvalidOperationException("COCO contains an image without file_name.");
            string[] directories = split == "val"
                ? new[] { "val", "valid", "val2017" }
                : new[] { split, split + "2017" };
            var candidates = new List<string> { Path.Combine(root, fileName) };
            foreach (string directory in directories)
                candidates.Add(Path.Combine(root, directory, fileName));
            candidates.Add(Path.Combine(root, "images", fileName));
            foreach (string candidate in candidates)
            {
                if (File.Exists(candidate))
                    return Path.GetFullPath(candidate);
            }

            string basename = Path.GetFileName(fileName);
            string discovered = Directory.EnumerateFiles(root, basename, SearchOption.AllDirectories)
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase).FirstOrDefault();
            if (discovered != null)
                return Path.GetFullPath(discovered);
            throw new FileNotFoundException("COCO image referenced by annotations was not found.", fileName);
        }

    }
}
