using System;

namespace OnnxVision.Imaging
{
    public enum OnnxPixelFormat
    {
        Bw8,
        Bgr24
    }

    public readonly struct OnnxImageBuffer
    {
        public OnnxImageBuffer(IntPtr data, int width, int height, int rowStride, OnnxPixelFormat pixelFormat)
        {
            Data = data;
            Width = width;
            Height = height;
            RowStride = rowStride;
            PixelFormat = pixelFormat;
        }

        public IntPtr Data { get; }
        public int Width { get; }
        public int Height { get; }
        public int RowStride { get; }
        public OnnxPixelFormat PixelFormat { get; }
        public int BytesPerPixel { get { return PixelFormat == OnnxPixelFormat.Bw8 ? 1 : 3; } }
    }
}
