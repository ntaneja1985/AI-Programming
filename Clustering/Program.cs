using Microsoft.ML;
using Microsoft.ML.Data;

public class CustomerData
{
    [LoadColumn(0)]
    public float CustomerID;

    [LoadColumn(1)]
    public float AnnualIncome;

    [LoadColumn(2)]
    public float SpendingScore;
}

public class ClusterPrediction
{
    [ColumnName("PredictedLabel")]
    public uint PredictedClusterId { get; set; }

    [ColumnName("Score")]
    public float[] Score { get; set; }

    public float CustomerID { get; set; }
}

public class Program
{
    private static readonly string dataPath = "customers.csv";

    public static void Main(string[] args)
    {
        var mlContext = new MLContext();    
        IDataView dataView = mlContext.Data.LoadFromTextFile<CustomerData>(dataPath, hasHeader: true, separatorChar: ',');
        //This is a transformation step in the pipeline. It concatenates the columns "AnnualIncome" and "SpendingScore" into a single column called "Features".
        //The Concatenate method is used to combine multiple columns into a single column, which is a common preprocessing step in machine learning.
        //The KMeans method is used to train a KMeans clustering model on the "Features" column.
        //The numberOfClusters parameter specifies the number of clusters to create.
        // K-means is an unsupervised machine learning algorithm used for clustering data points into a specified number of clusters.
        var pipeline = mlContext.Transforms.Concatenate("Features", "AnnualIncome", "SpendingScore")
            .Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3));
        var model = pipeline.Fit(dataView);
        //The Transform method is used to apply the trained model to the input data and generate predictions.
        var transformedData = model.Transform(dataView);
        //The CreateEnumerable method is used to extract the predictions from the transformed data.
        var predictions = mlContext.Data.CreateEnumerable<ClusterPrediction>(transformedData, reuseRowObject: false);
        foreach (var prediction in predictions)
        {
            Console.WriteLine($"Customer: {prediction.CustomerID} - Cluster: {prediction.PredictedClusterId}");
        }

    }
}