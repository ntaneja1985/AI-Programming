using Microsoft.ML;
using Microsoft.ML.Data;


MLContext mlContext = new MLContext();
IDataView data = mlContext.Data.LoadFromTextFile<HousingData>("housing-data.csv", separatorChar: ',');

// Define the data preparation pipeline
// Convert the SquareFeet column to a Single type
// Normalize the SquareFeet column
// Concatenate the SquareFeet and Bedrooms columns into a Features column
// One-hot encode the Neighborhood column

var dataPipeline = 
    mlContext.Transforms.Conversion.ConvertType("SquareFeet", outputKind: DataKind.Single)
    .Append(mlContext.Transforms.NormalizeMinMax("SquareFeet"))
    .Append(mlContext.Transforms.Concatenate("Features", "SquareFeet", "Bedrooms"))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"));

// Fit and transform the data
var transformedData = dataPipeline.Fit(data).Transform(data);

// Create an enumerable of the transformed data
var transformedDataEnumerable = mlContext.Data.CreateEnumerable<TransformedHousingData>(transformedData, reuseRowObject: false).ToList();

foreach (var item in transformedDataEnumerable)
{
    Console.WriteLine($"SquareFeet: {item.SquareFeet}," +
        $" Bedrooms: {item.Bedrooms}, " +
        $"Price: {item.Price}, " +
        $"Features: [{string.Join(", ", item.Features)}], " +
        $"Neighborhood: [{string.Join(", ", item.Neighborhood)}]");
}

//string[] featureColumns = { "SquareFeet", "Bedrooms" };
//string labelColumn = "Price";

    //// Define the training pipeline
    //var pipeline = mlContext.Transforms.Concatenate("Features", featureColumns)
    //    .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: labelColumn));

    //// Train the model
    //var model = pipeline.Fit(data);

    //// Make predictions
    //var prediction = model.Transform(data);

    //// Evaluate the model
    //var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: labelColumn);

    //// Print the evaluation metrics
    //Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
    //Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");


    // Define a class to hold the housing data
public class HousingData
{
    [LoadColumn(0)]
    public float SquareFeet { get; set; }

    [LoadColumn(1)]
    public float Bedrooms { get; set; }

    [LoadColumn(2)]
    public float Price { get; set; }

    [LoadColumn(3)]
    public string Neighborhood { get; set; }
}

public class TransformedHousingData
{
    public float SquareFeet { get; set; }
    public float Bedrooms { get; set; }
    public float Price { get; set; }
    public float[] Features { get; set; }
    public float[] Neighborhood { get; set; }
}

// Define a class to hold the housing prediction
public class HousingPrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}