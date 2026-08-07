using System;
using System.Drawing;
using System.Globalization;
using OnnxVision.Runtime;

namespace OnnxVision
{
    internal static partial class Program
    {
        private static bool TryParseClassificationArguments(string[] args, int offset,
            int defaultRepeats, out OnnxExecutionProvider[] providers, out int repeats,
            out RoiPlacement roi, out InputOptions inputOptions)
        {
            providers = new[] { OnnxExecutionProvider.Cpu };
            repeats = defaultRepeats;
            roi = null;
            inputOptions = new InputOptions();

            int index = offset + 2;
            int repeatArguments = 0;
            int providerArguments = 0;
            while (index < args.Length)
            {
                if (TryParseInputOption(args, ref index, inputOptions))
                    continue;

                if (IsFlag(args[index], "roi"))
                {
                    if (roi != null || index + 4 >= args.Length ||
                        !TryParseRoi(args, index + 1, out roi))
                    {
                        return false;
                    }
                    index += 5;
                    continue;
                }

                if (args.Length - index >= 4 && TryParseRoi(args, index, out roi))
                {
                    index += 4;
                    continue;
                }

                OnnxExecutionProvider[] parsedProviders;
                if (TryParseProvider(args[index], out parsedProviders))
                {
                    if (++providerArguments > 1)
                        return false;
                    providers = parsedProviders;
                    index++;
                    continue;
                }

                int parsedRepeats;
                if (!TryParsePositiveInteger(args[index], out parsedRepeats) || ++repeatArguments > 1)
                    return false;
                repeats = parsedRepeats;
                index++;
            }

            return repeatArguments <= 1 && providerArguments <= 1;
        }

        private static bool TryParseDetectionArguments(string[] args, int offset,
            int defaultRepeats, out float threshold, out int repeats,
            out OnnxExecutionProvider[] providers, out InputOptions inputOptions)
        {
            threshold = 0.5f;
            repeats = defaultRepeats;
            providers = new[] { OnnxExecutionProvider.Cpu };
            inputOptions = new InputOptions();
            bool thresholdSpecified = false;
            bool repeatsSpecified = false;
            bool providerSpecified = false;

            int index = offset + 2;
            while (index < args.Length)
            {
                if (TryParseInputOption(args, ref index, inputOptions))
                    continue;

                OnnxExecutionProvider[] parsedProviders;
                if (TryParseProvider(args[index], out parsedProviders))
                {
                    if (providerSpecified)
                        return false;
                    providers = parsedProviders;
                    providerSpecified = true;
                    index++;
                    continue;
                }

                float parsedThreshold;
                if (!thresholdSpecified && TryParseThreshold(args[index], out parsedThreshold))
                {
                    threshold = parsedThreshold;
                    thresholdSpecified = true;
                    index++;
                    continue;
                }

                int parsedRepeats;
                if (!repeatsSpecified && TryParsePositiveInteger(args[index], out parsedRepeats))
                {
                    repeats = parsedRepeats;
                    repeatsSpecified = true;
                    index++;
                    continue;
                }

                return false;
            }

            return true;
        }

        private static bool TryParseInputOption(string[] args, ref int index,
            InputOptions inputOptions)
        {
            string value = args[index];
            string batchSizeValue;
            if (TryParseInlineOption(value, "batch-size", out batchSizeValue))
            {
                int batchSize;
                if (inputOptions.BatchSize.HasValue ||
                    !TryParsePositiveInteger(batchSizeValue, out batchSize))
                {
                    return false;
                }
                inputOptions.BatchSize = batchSize;
                index++;
                return true;
            }

            if (IsFlag(value, "batch-size"))
            {
                int batchSize;
                if (inputOptions.BatchSize.HasValue || index + 1 >= args.Length ||
                    !TryParsePositiveInteger(args[index + 1], out batchSize))
                {
                    return false;
                }
                inputOptions.BatchSize = batchSize;
                index += 2;
                return true;
            }

            if (IsFlag(value, "validate"))
            {
                inputOptions.Validate = true;
                index++;
                return true;
            }

            if (IsFlag(value, "dataset"))
            {
                inputOptions.ForceDataset = true;
                index++;
                return true;
            }

            string set;
            if (TryParseInlineOption(value, "set", out set))
            {
                if (!TrySetDatasetSplit(inputOptions, set))
                    return false;
                index++;
                return true;
            }

            if (IsFlag(value, "set"))
            {
                if (index + 1 >= args.Length ||
                    !TrySetDatasetSplit(inputOptions, args[index + 1]))
                {
                    return false;
                }
                index += 2;
                return true;
            }

            return false;
        }

        private static bool TrySetDatasetSplit(InputOptions inputOptions, string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return false;
            string split = value.Trim().ToLowerInvariant();
            if (split == "valid" || split == "validation")
                split = "val";
            if (split != "train" && split != "val" && split != "test")
                return false;
            if (inputOptions.Set != null)
                return false;
            inputOptions.Set = split;
            return true;
        }

        private static bool TryParseInlineOption(string value, string name, out string optionValue)
        {
            optionValue = null;
            string prefix = "-" + name + "=";
            string longPrefix = "--" + name + "=";
            if (value.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                optionValue = value.Substring(prefix.Length);
            else if (value.StartsWith(longPrefix, StringComparison.OrdinalIgnoreCase))
                optionValue = value.Substring(longPrefix.Length);
            else
                return false;
            return true;
        }

        private static bool IsFlag(string value, string name)
        {
            return string.Equals(value, "-" + name, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(value, "--" + name, StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsJsonFlag(string value)
        {
            return IsFlag(value, "json");
        }

        private static bool TryParseProvider(
            string value, out OnnxExecutionProvider[] providers)
        {
            if (string.IsNullOrWhiteSpace(value) ||
                string.Equals(value, "cpu", StringComparison.OrdinalIgnoreCase))
            {
                providers = new[] { OnnxExecutionProvider.Cpu };
                return true;
            }
            providers = null;
            return false;
        }

        private static bool TryParseThreshold(string value, out float threshold)
        {
            threshold = 0.5f;
            if (string.IsNullOrWhiteSpace(value))
                return true;
            return float.TryParse(value, NumberStyles.Float,
                CultureInfo.InvariantCulture, out threshold) &&
                threshold >= 0 && threshold <= 1;
        }

        private static bool TryParsePositiveInteger(string value, out int result)
        {
            return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out result) &&
                result > 0;
        }

        private static bool TryParseRoi(string[] args, int offset, out RoiPlacement placement)
        {
            placement = null;
            int x;
            int y;
            int width;
            int height;
            if (!int.TryParse(args[offset], out x) ||
                !int.TryParse(args[offset + 1], out y) ||
                !int.TryParse(args[offset + 2], out width) ||
                !int.TryParse(args[offset + 3], out height) ||
                width <= 0 || height <= 0)
            {
                return false;
            }

            placement = new RoiPlacement(x, y, width, height);
            return true;
        }

        private sealed class InputOptions
        {
            public bool Validate { get; set; }
            public bool ForceDataset { get; set; }
            public string Set { get; set; }
            public int? BatchSize { get; set; }
        }

        private sealed class RoiPlacement
        {
            public RoiPlacement(int x, int y, int width, int height)
            {
                X = x;
                Y = y;
                Width = width;
                Height = height;
            }

            public int X { get; private set; }
            public int Y { get; private set; }
            public int Width { get; private set; }
            public int Height { get; private set; }

            public Rectangle ToRectangle()
            {
                return new Rectangle(X, Y, Width, Height);
            }

            public override string ToString()
            {
                return string.Format(CultureInfo.InvariantCulture,
                    "ROI ({0}, {1}, {2}, {3})", X, Y, Width, Height);
            }
        }
    }
}
