using System;
using System.Collections.Generic;
using System.Linq;
using System.Web.Script.Serialization;
using Microsoft.ML.OnnxRuntime;

namespace OnnxVision.Runtime
{
    internal sealed class OnnxContractMetadata
    {
        private OnnxContractMetadata(string[] classNames, string inputVariant,
            bool dynamicBatch, int? fixedBatchSize, bool nmsRequired)
        {
            ClassNames = classNames;
            InputVariant = inputVariant;
            DynamicBatch = dynamicBatch;
            FixedBatchSize = fixedBatchSize;
            NmsRequired = nmsRequired;
        }

        public string[] ClassNames { get; private set; }
        public string InputVariant { get; private set; }
        public bool DynamicBatch { get; private set; }
        public int? FixedBatchSize { get; private set; }
        public bool NmsRequired { get; private set; }

        public static OnnxContractMetadata Read(InferenceSession session,
            string expectedTask, string expectedName)
        {
            var metadata = session.ModelMetadata.CustomMetadataMap;
            RequireEqual(metadata, "vision_task", expectedTask);
            RequireEqual(metadata, "contract_name", expectedName);
            ValidateVersion(Require(metadata, "contract_version"));
            string variant = Require(metadata, "input_variant");
            if (variant != "bw8" && variant != "c24")
                throw new NotSupportedException("input_variant must be bw8 or c24.");

            string[] names = ParseNames(ParseObject(metadata, "names"));
            bool dynamic;
            int? fixedSize;
            ValidateInputs(ParseObject(metadata, "inputs"), variant, out dynamic, out fixedSize);
            var outputs = ParseObject(metadata, "outputs");
            bool nmsRequired = false;
            if (expectedTask == OnnxVisionContract.ClassificationTask)
                ValidateClassificationOutputs(outputs);
            else
            {
                ValidateDetectionOutputs(outputs);
                string serializedNms = Require(metadata, "nms_required");
                if (serializedNms != "true" && serializedNms != "false")
                    throw new NotSupportedException("nms_required must be a boolean.");
                nmsRequired = serializedNms == "true";
            }
            ValidateTensors(session, expectedTask, variant, names.Length, dynamic, fixedSize);
            return new OnnxContractMetadata(names, variant, dynamic, fixedSize, nmsRequired);
        }

        private static void ValidateVersion(string value)
        {
            string[] parts = value.Split('.');
            int major, minor, micro;
            if (parts.Length != 3 || !int.TryParse(parts[0], out major) ||
                !int.TryParse(parts[1], out minor) || !int.TryParse(parts[2], out micro) ||
                major != OnnxVisionContract.MajorVersion || minor < 0 || micro < 0)
                throw new NotSupportedException("Expected a compatible contract version 2.x.y.");
        }

        private static void ValidateInputs(IDictionary<string, object> inputs, string variant,
            out bool dynamic, out int? fixedSize)
        {
            var batch = Object(inputs, "batch");
            if (Integer(batch, "axis") != 0)
                throw new NotSupportedException("The batch axis must be zero.");
            string mode = String(batch, "mode");
            if (mode == "dynamic")
            {
                if (Integer(batch, "minimum") != 1)
                    throw new NotSupportedException("Dynamic batch minimum must be one.");
                dynamic = true;
                fixedSize = null;
            }
            else if (mode == "fixed")
            {
                int size = Integer(batch, "size");
                if (size <= 0)
                    throw new NotSupportedException("Fixed batch size must be positive.");
                dynamic = false;
                fixedSize = size;
            }
            else
                throw new NotSupportedException("Batch mode must be dynamic or fixed.");

            var variants = Object(inputs, "variants");
            ValidateVariant(Object(variants, "bw8"), "images_bw8_uint8_nchw",
                "NCHW", "BW8", new object[] { "B", 1, "H", "W" });
            ValidateVariant(Object(variants, "c24"), "images_c24_uint8_nhwc_bgr",
                "NHWC", "C24_BGR", new object[] { "B", "H", "W", 3 });
            Object(variants, variant);
        }

        private static void ValidateVariant(IDictionary<string, object> value, string name,
            string layout, string pixelFormat, object[] shape)
        {
            Equal(value, "name", name);
            Equal(value, "dtype", "uint8");
            Equal(value, "layout", layout);
            Equal(value, "pixel_format", pixelFormat);
            Equal(value, "preprocessing", "embedded");
            Shape(value, shape);
        }

        private static void ValidateClassificationOutputs(IDictionary<string, object> outputs)
        {
            var probabilities = Object(outputs, "probabilities");
            Equal(probabilities, "dtype", "float32");
            Shape(probabilities, new object[] { "B", "C" });
            Equal(probabilities, "semantics", "categorical_probabilities");
            NumberArray(probabilities, "range", 0.0, 1.0);
            Number(probabilities, "row_sum", 1.0);
        }

        private static void ValidateDetectionOutputs(IDictionary<string, object> outputs)
        {
            var boxes = Object(outputs, "boxes");
            Equal(boxes, "dtype", "float32");
            Shape(boxes, new object[] { "B", "Q", 4 });
            Equal(boxes, "coordinate_format", "xyxy");
            Equal(boxes, "coordinate_space", "normalized_0_1");
            var scores = Object(outputs, "scores");
            Equal(scores, "dtype", "float32");
            Shape(scores, new object[] { "B", "Q" });
            NumberArray(scores, "range", 0.0, 1.0);
            Number(scores, "padding_value", 0.0);
            Equal(scores, "valid_when", "score_gt_0");
            var classIds = Object(outputs, "class_ids");
            Equal(classIds, "dtype", "int64");
            Shape(classIds, new object[] { "B", "Q" });
        }

        private static void ValidateTensors(InferenceSession session, string task,
            string variant, int classes, bool dynamic, int? fixedSize)
        {
            if (session.InputMetadata.Count != 1)
                throw new NotSupportedException("The contract requires exactly one input tensor.");
            var input = session.InputMetadata.Single();
            string inputName = variant == "bw8" ? "images_bw8_uint8_nchw" : "images_c24_uint8_nhwc_bgr";
            if (input.Key != inputName || input.Value.ElementType != typeof(byte) ||
                input.Value.Dimensions.Length != 4)
                throw new NotSupportedException("The input tensor does not match input_variant.");
            Batch(input.Value.Dimensions[0], dynamic, fixedSize, "input");
            if ((variant == "bw8" && input.Value.Dimensions[1] != 1) ||
                (variant == "c24" && input.Value.Dimensions[3] != 3))
                throw new NotSupportedException("The input channel dimension is invalid.");

            if (task == OnnxVisionContract.ClassificationTask)
            {
                if (session.OutputMetadata.Count != 1 || !session.OutputMetadata.ContainsKey("probabilities"))
                    throw new NotSupportedException("Classification requires exactly one probabilities output.");
                NodeMetadata output = session.OutputMetadata["probabilities"];
                if (output.ElementType != typeof(float) || output.Dimensions.Length != 2 ||
                    output.Dimensions[1] != classes)
                    throw new NotSupportedException("Classification probabilities must be float32 [B,C].");
                Batch(output.Dimensions[0], dynamic, fixedSize, "probabilities");
                return;
            }

            string[] names = { "boxes", "scores", "class_ids" };
            if (session.OutputMetadata.Count != 3 || names.Any(name => !session.OutputMetadata.ContainsKey(name)))
                throw new NotSupportedException("Detection requires exactly boxes, scores, and class_ids outputs.");
            Output(session.OutputMetadata["boxes"], typeof(float), 3, "boxes");
            Output(session.OutputMetadata["scores"], typeof(float), 2, "scores");
            Output(session.OutputMetadata["class_ids"], typeof(long), 2, "class_ids");
            if (session.OutputMetadata["boxes"].Dimensions[2] != 4)
                throw new NotSupportedException("Detection boxes must end with dimension four.");
            int queryCount = session.OutputMetadata["boxes"].Dimensions[1];
            if (session.OutputMetadata["scores"].Dimensions[1] != queryCount ||
                session.OutputMetadata["class_ids"].Dimensions[1] != queryCount)
                throw new NotSupportedException("Detection output query dimensions must agree.");
            foreach (string name in names)
                Batch(session.OutputMetadata[name].Dimensions[0], dynamic, fixedSize, name);
        }

        private static void Output(NodeMetadata value, Type type, int rank, string name)
        {
            if (value.ElementType != type || value.Dimensions.Length != rank)
                throw new NotSupportedException("Invalid detection output tensor: " + name + ".");
        }

        private static void Batch(int actual, bool dynamic, int? fixedSize, string tensor)
        {
            if (dynamic ? actual > 0 : actual != fixedSize.Value)
                throw new NotSupportedException("Batch dimension does not match metadata for " + tensor + ".");
        }

        private static string[] ParseNames(IDictionary<string, object> mapping)
        {
            if (mapping.Count == 0)
                throw new NotSupportedException("The class mapping must not be empty.");
            var names = new string[mapping.Count];
            foreach (var item in mapping)
            {
                int index;
                string name = item.Value as string;
                if (!int.TryParse(item.Key, out index) || index < 0 || index >= names.Length ||
                    names[index] != null || string.IsNullOrWhiteSpace(name))
                    throw new NotSupportedException("Class indices must be contiguous and start at zero.");
                names[index] = name;
            }
            if (names.Any(name => name == null))
                throw new NotSupportedException("Class indices must be contiguous and start at zero.");
            return names;
        }

        private static IDictionary<string, object> ParseObject(
            IReadOnlyDictionary<string, string> metadata, string key)
        {
            try
            {
                var value = new JavaScriptSerializer().DeserializeObject(Require(metadata, key))
                    as IDictionary<string, object>;
                if (value == null)
                    throw new NotSupportedException("Contract metadata must be an object: " + key + ".");
                return value;
            }
            catch (ArgumentException error)
            {
                throw new NotSupportedException("Invalid JSON contract metadata: " + key + ".", error);
            }
        }

        private static IDictionary<string, object> Object(IDictionary<string, object> value, string key)
        {
            object child;
            var result = value.TryGetValue(key, out child) ? child as IDictionary<string, object> : null;
            if (result == null)
                throw new NotSupportedException("Missing object contract field: " + key + ".");
            return result;
        }

        private static string String(IDictionary<string, object> value, string key)
        {
            object item;
            string result = value.TryGetValue(key, out item) ? item as string : null;
            if (result == null)
                throw new NotSupportedException("Missing string contract field: " + key + ".");
            return result;
        }

        private static int Integer(IDictionary<string, object> value, string key)
        {
            object item;
            if (!value.TryGetValue(key, out item))
                throw new NotSupportedException("Missing integer contract field: " + key + ".");
            try { return Convert.ToInt32(item); }
            catch (Exception error) when (error is FormatException || error is InvalidCastException || error is OverflowException)
            { throw new NotSupportedException("Invalid integer contract field: " + key + ".", error); }
        }

        private static void Shape(IDictionary<string, object> value, object[] expected)
        {
            object raw;
            object[] actual = value.TryGetValue("shape", out raw) ? raw as object[] : null;
            if (actual == null || actual.Length != expected.Length)
                throw new NotSupportedException("Invalid tensor shape metadata.");
            for (int index = 0; index < expected.Length; index++)
                if (Convert.ToString(actual[index]) != Convert.ToString(expected[index]))
                    throw new NotSupportedException("Invalid tensor shape metadata.");
        }

        private static void NumberArray(IDictionary<string, object> value, string key,
            double first, double second)
        {
            object raw;
            object[] values = value.TryGetValue(key, out raw) ? raw as object[] : null;
            if (values == null || values.Length != 2 || Convert.ToDouble(values[0]) != first ||
                Convert.ToDouble(values[1]) != second)
                throw new NotSupportedException("Invalid numeric contract field: " + key + ".");
        }

        private static void Number(IDictionary<string, object> value, string key, double expected)
        {
            object actual;
            if (!value.TryGetValue(key, out actual) || Convert.ToDouble(actual) != expected)
                throw new NotSupportedException("Invalid numeric contract field: " + key + ".");
        }

        private static void Equal(IDictionary<string, object> value, string key, string expected)
        {
            if (String(value, key) != expected)
                throw new NotSupportedException("Invalid contract field: " + key + ".");
        }

        private static string Require(IReadOnlyDictionary<string, string> metadata, string key)
        {
            string value;
            if (!metadata.TryGetValue(key, out value) || string.IsNullOrWhiteSpace(value))
                throw new NotSupportedException("Missing required ONNX metadata: " + key + ".");
            return value;
        }

        private static void RequireEqual(IReadOnlyDictionary<string, string> metadata,
            string key, string expected)
        {
            if (!string.Equals(Require(metadata, key), expected, StringComparison.Ordinal))
                throw new NotSupportedException("Invalid ONNX metadata: " + key + ".");
        }
    }
}
