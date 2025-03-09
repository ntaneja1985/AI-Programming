using Microsoft.ML;
using Microsoft.ML.Data;

namespace NetworkTrafficAnomalyDetection
{

    public class NetworkTrafficData
    {
        [LoadColumn(0)]
        public string Timestamp { get; set; }

        [LoadColumn(1)]
        public string SourceIP { get; set; }

        [LoadColumn(2)]
        public string DestinationIP { get; set; }

        [LoadColumn(3)]
        public string Protocol { get; set; }

        [LoadColumn(4)]
        public float PacketSize { get; set; }

        [LoadColumn(5)]
        public string Label { get; set; }

    }

    public class NetworkTrafficPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId { get; set; }

        public float[] Score { get; set; }
    }
    class Program
    {
        static void Main(string[] args)
        {
            var mLContext = new MLContext();
            var dataPath = "network_data.csv";
            var dataView = mLContext.Data.LoadFromTextFile<NetworkTrafficData>(dataPath, hasHeader: true, separatorChar: ',');
            //var preview = dataView.Preview();
            //foreach (var row in preview.RowView)
            //{
            //    Console.WriteLine($"{row.Values[0]} | {row.Values[1]}");
            //}


            /*
             In the first step of the pipeline convert the "SourceIP" column in the data to a numeric key. 
              This is useful when working with categorical data in machine learning models.
             Convert the "DestinationIP" column in the data to a numeric key, similar to the previous step.
             Concatenate the "Features" column with the "PacketSize" column. 
             The "Features" column is a combination of multiple input features that will be used for training the model.
             Normalize the values in the "Features" column. 
             Normalization is a common preprocessing step in machine learning that scales the values to a specific range, often between 0 and 1. 
             This ensures that all features have a similar impact on the model.
             Apply the K-means clustering algorithm to the "Features" column. 
             K-means is an unsupervised machine learning algorithm that groups similar data points together. 
            In this case, it will cluster the data into 3 groups based on the values in the "Features" column.
             */
            var pipeline = mLContext.Transforms.Conversion.MapValueToKey("SourceIP")
                .Append(mLContext.Transforms.Conversion.MapValueToKey("DestinationIP"))
                .Append(mLContext.Transforms.Concatenate("Features","PacketSize"))
                .Append(mLContext.Transforms.NormalizeMinMax("Features"))
                .Append(mLContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));

            // Train the model
            var model = pipeline.Fit(dataView);

            // Make predictions
            var predictions = model.Transform(dataView);

            var predictedData = mLContext.Data.CreateEnumerable<NetworkTrafficPrediction>(predictions, reuseRowObject: false);
            var actualData = mLContext.Data.CreateEnumerable<NetworkTrafficData>(dataView, reuseRowObject: false);

            using (var predictedEnumerator = predictedData.GetEnumerator())
            using (var actualEnumerator = actualData.GetEnumerator())
            {
                while (predictedEnumerator.MoveNext() && actualEnumerator.MoveNext())
                {
                    var predicted = predictedEnumerator.Current;
                    var actual = actualEnumerator.Current;

                    var predictedLabel = predicted.PredictedClusterId == 1? "Normal": "Anomalous";
                    Console.WriteLine($"Actual Label: {actual.Label}, Predicted Label: {predictedLabel}, Score:{string.Join(", ",predicted.Score)}");
                }
            }

            Console.WriteLine("Anomaly Detection Complete.");
        }
    }
}