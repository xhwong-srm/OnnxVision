namespace OnnxVision
{
    internal static partial class Program
    {
        private sealed class RocPoint
        {
            public RocPoint(bool actualPositive, float score)
            {
                ActualPositive = actualPositive;
                Score = score;
            }

            public bool ActualPositive { get; private set; }
            public float Score { get; private set; }
        }
    }
}
