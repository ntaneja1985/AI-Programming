using Microsoft.ML;
using Microsoft.ML.Data;


namespace HousePricePrediction
{
    public class HouseData
    {
        [LoadColumn(0)]
        public float HouseSizeSqft { get; set; }

        [LoadColumn(1)]
        public float NumBedrooms { get; set; }

        [LoadColumn(2)]
        public float NumBathrooms { get; set; }

        [LoadColumn(3)]
        public string Neighborhood { get; set; }

        [LoadColumn(4)]
        public float SalePrice { get; set; }    
    }

    public class HousePrediction
    {
        [ColumnName("Score")]
        public float PredictedSalePrice { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);
            var dataPath = Path.Combine(Environment.CurrentDirectory, "house-price-data.csv");

            /*
             Specify the path to the data file containing the house price data and loads it into an IDataView object using the LoadFromTextFile method.
             */
            IDataView data = mlContext.Data.LoadFromTextFile<HouseData>(dataPath, hasHeader: true, separatorChar: ',');

            /*
             The TrainTestSplit method is used to split the data into a training set and a test set. The testFraction parameter specifies the fraction of the data that should be used for testing.
             Data is split into training and testing datasets using the TrainTestSplit method, with 80% of the data used for training and 20% for testing.
             */
            var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            /*
             The pipeline is defined using the Concatenate, OneHotEncoding, CopyColumns, and FastTreeRegression classes. 
            The pipeline is used to concatenate the features into a single column, one-hot encode the neighborhood column, and copy the sale price column to the label column. 
            The FastTreeRegression class is used to train the model.
             */
            var pipeline = mlContext.Transforms.Concatenate("Features", "HouseSizeSqft", "NumBedrooms", "NumBathrooms")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"))
                .Append(mlContext.Transforms.Concatenate("Features", "Features", "Neighborhood"))
                .Append(mlContext.Transforms.CopyColumns("Label", "SalePrice"))
                .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "Label"));

            /*
             The Fit method is used to train the model using the training data.
             */
            var trainedModel = pipeline.Fit(trainData);

            /*
             The Transform method is used to make predictions on the test data.
             */
            var predictions = trainedModel.Transform(testData);

            /*
             The Evaluate method is used to evaluate the model using the test data and calculate the RSquared score and root mean squared error.
             */
            var metrics = mlContext.Regression.Evaluate(predictions);

            //Console.WriteLine($"RSquared Score: {metrics.RSquared:0.##}");
            //Console.WriteLine($"Root Mean Squared error: {metrics.RootMeanSquaredError:0.##}");


            /*
             The CreatePredictionEngine method is used to create a prediction engine for making predictions on new data.
             */
            var predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(trainedModel);

            var houseData = new HouseData()
            {
                HouseSizeSqft = 2000,
                NumBedrooms = 3,
                NumBathrooms = 2,
                Neighborhood = "Southwest"
            };

            /*
             The Predict method is used to make a prediction using the prediction engine.
             */
            var prediction = predictionEngine.Predict(houseData);
            Console.WriteLine($"Predicted Sale Price: ${prediction.PredictedSalePrice}");
        }
    }
}