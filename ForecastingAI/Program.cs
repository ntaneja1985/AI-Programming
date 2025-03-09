using Microsoft.ML;
using Microsoft.ML.Data;

namespace StockPriceForecasting
{
    class Program
    {
        public class StockData
        {
            [LoadColumn(0)]
            public string Date { get; set; }

            [LoadColumn(1)]
            public float Open { get; set; }

            [LoadColumn(2)]
            public float High { get; set; }

            [LoadColumn(3)]
            public float Low { get; set; }

            [LoadColumn(4)]
            public float Close { get; set; }
        }

        public class StockPrediction
        {
            [ColumnName("Score")]
            public float PredictedClose { get; set; }
        }

            static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            IDataView dataView = mlContext.Data.LoadFromTextFile<StockData>("stock_data.csv", hasHeader: true, separatorChar: ',');

            var preview = dataView.Preview();

            foreach (var row in preview.RowView)
            {
                Console.WriteLine($"{row.Values[0]} | {row.Values[1]}");
            }

            /*
             1.	mlContext.Transforms.Concatenate("Features", "Open", "High", "Low"): 
                This transformation concatenates the "Open", "High", and "Low" columns of the input data and creates a new column called "Features". 
                The "Features" column will be used as input for the regression trainer.
             2.	.Append(mlContext.Transforms.CopyColumns("Label", "Close")): 
                This transformation copies the values from the "Close" column of the input data and creates a new column called "Label". 
                The "Label" column represents the target variable that the regression trainer will try to predict.
             3.	.Append(mlContext.Regression.Trainers.FastTree()): 
                This appends the regression trainer to the pipeline. 
                In this case, the FastTree regression trainer is used. 
                The regression trainer will use the "Features" column as input and the "Label" column as the target variable to train a machine learning model.
             */
            var pipeline = mlContext.Transforms.Concatenate("Features", "Open", "High", "Low")
                            .Append(mlContext.Transforms.CopyColumns("Label", "Close"))
                            .Append(mlContext.Regression.Trainers.FastTree());

            var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var model = pipeline.Fit(trainTestData.TrainSet);

            /*
             Apply the trained model to the test dataset and generates predictions for the target variable. 
            These predictions can be used to evaluate the accuracy of the machine learning model and compare them with the actual closing prices of the stocks.
             */
            var predictions = model.Transform(trainTestData.TestSet);

            // Evaluate the model
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine($"R-Squared: {metrics.RSquared}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

            var predictedResult = mlContext.Data.CreateEnumerable<StockPrediction>(predictions, reuseRowObject: false).ToList();

            var testData = mlContext.Data.CreateEnumerable<StockData>(trainTestData.TestSet, reuseRowObject: false).ToList();

            /*
               Iterate over two collections simultaneously: predictedResult and testData. 
               It uses the Zip method to combine the elements of both collections into tuples (prediction, actual).
               For each pair of elements, the loop prints the predicted and actual values of the stock's closing price using the Console.WriteLine method. 
               The predicted value is accessed through the prediction.PredictedClose property, and the actual value is accessed through the actual.Close property.
               In other words, this loop is used to compare the predicted closing prices of stocks with their actual closing prices. It can be helpful for evaluating the accuracy of the machine learning model used to make the predictions.

             */
            foreach (var (prediction, actual) in predictedResult.Zip(testData, (p, a) => (p, a)))
            {
                Console.WriteLine($"Predicted: {prediction.PredictedClose}, Actual: {actual.Close}");
            }

        }

    }
}