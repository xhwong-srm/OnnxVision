using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace OnnxVision.Runtime
{
    public static class OnnxRuntimeEnvironment
    {
        private static readonly string[] RequiredRuntimeDlls =
        {
            "Microsoft.ML.OnnxRuntime.dll",
            "onnxruntime.dll",
            "onnxruntime_providers_shared.dll"
        };

        private static readonly List<IntPtr> VisualCppRuntimeHandles = new List<IntPtr>();
        private static readonly object VisualCppRuntimeLock = new object();

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr LoadLibrary(string lpFileName);

        public static bool ValidateDeployment(string baseDirectory, out string errorMessage)
        {
            if (string.IsNullOrWhiteSpace(baseDirectory))
                throw new ArgumentException("ONNX runtime base directory is required.", "baseDirectory");

            var missingDlls = new List<string>();
            foreach (string dllName in RequiredRuntimeDlls)
            {
                if (!File.Exists(Path.Combine(baseDirectory, dllName)))
                    missingDlls.Add(dllName);
            }

            errorMessage = missingDlls.Count == 0
                ? ""
                : "Missing Microsoft ONNX Runtime DLLs: " + string.Join(", ", missingDlls);
            return missingDlls.Count == 0;
        }

        public static bool TryPreloadVisualCppRuntime(out string errorMessage)
        {
            lock (VisualCppRuntimeLock)
            {
                if (VisualCppRuntimeHandles.Count > 0)
                {
                    errorMessage = "";
                    return true;
                }

                string[] runtimeDlls =
                {
                    "VCRUNTIME140.dll",
                    "VCRUNTIME140_1.dll",
                    "MSVCP140.dll",
                    "MSVCP140_1.dll"
                };

                foreach (string runtimeDll in runtimeDlls)
                {
                    string runtimePath = Path.Combine(Environment.SystemDirectory, runtimeDll);
                    IntPtr handle = LoadLibrary(runtimePath);
                    if (handle == IntPtr.Zero)
                    {
                        errorMessage = "Failed to load the required Microsoft Visual C++ runtime:\n" +
                            runtimePath + "\n\nPlease install the latest Microsoft Visual C++ Redistributable (x64).";
                        return false;
                    }

                    VisualCppRuntimeHandles.Add(handle);
                }
            }

            errorMessage = "";
            return true;
        }
    }
}
