using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using Microsoft.ML.Trainers;
using System.Runtime.CompilerServices;
using System.ComponentModel;
using System.Reflection.Emit;

namespace Classification
{
    public class MovieReview
    {
        //The load column attribute is used in Ml.net to specify the index of the column in a data set to load into a property of a class when loading data from a file.
        //In this case, the attribute will load values from the first column index zero of the dataset file into the label property.

        [LoadColumn(0)]
        public string text { get; set; }

        [LoadColumn(1)]
        [ColumnName("Label")]
        public bool sentiment { get; set; }


    }

    public class TextData
    {
        [LoadColumn(0)]
        public string text { get; set; }
    }

    public class SentimentPrediction
    {
        [ColumnName("Score")]
        public float SentimentScore { get; set; }

        public bool IsPositiveSentiment => SentimentScore < 0.5f;
    }

        public class Program
    {
        public static void Main(string[] args)
        {
            ////Create a new MLContext instance
            //MLContext mLContext = new MLContext();
            ////Load the data from the file into an IDataView object
            //string dataPath = "train.csv";
            ////string text = File.ReadAllText(dataPath);
            ////using (StreamReader streamReader = new StreamReader(dataPath))
            ////{
            ////    text = text.Replace("\'", "");
            ////} 

            ////File.WriteAllText(dataPath, text);


            //IDataView dataView = mLContext.Data.LoadFromTextFile<MovieReview>(dataPath, hasHeader: true, allowQuoting: true, separatorChar: ',');

            ////Console.WriteLine("Data loaded successfully");
            ////Console.WriteLine();

            ////var preview = dataView.Preview();
            ////foreach (var row in preview.RowView)
            ////{
            ////    Console.WriteLine($"{row.Values[0]} | {row.Values[1]}");
            ////}

            ////The pipeline starts with the Transforms.Text.FeaturizeText method.
            ////This method is used to convert the text data into numerical features that can be used by the machine learning algorithm.
            ////It takes two parameters: the name of the output column ("Features") and the name of the input column ("text").
            ////The Append method is then called on the pipeline to add another component to the sequence.
            ////In this case, it appends the BinaryClassification.Trainers.SdcaLogisticRegression method, which represents the chosen machine learning algorithm.
            ////This algorithm is a binary logistic regression model trained using the Stochastic Dual Coordinate Ascent(SDCA) optimization algorithm.
            ////It takes two parameters: the name of the label column("Label") and the name of the feature column("Features").
            ////in summary, this line of code creates a pipeline that first converts the text data into numerical features and then applies a binary logistic regression model to train the data.
            //var pipeline = mLContext.Transforms.Text.FeaturizeText("Features", "text")
            //    .Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));

            //    //In machine learning, a pipeline is a sequence of data processing components, called transformers and estimators, that are applied in a specific order to transform the data and train a model.
            //    //The Fit method is used to train the model by fitting the pipeline to the data.
            //    //In this case, the pipeline variable represents the sequence of transformations and the chosen machine learning algorithm.
            //    //The Fit method takes the dataView as input and trains the model by applying the transformations and the chosen algorithm to the data.
            //var model = pipeline.Fit(dataView);

            ////The Transform method is used to apply the trained model to new data and generate predictions.
            //var predictions = model.Transform(dataView);

            ////The Evaluate method is used to evaluate the model's performance on the test data.
            //var metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");
            //Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            //Console.WriteLine($"Accuracy: {metrics.PositivePrecision}");
            //Console.WriteLine($"Accuracy: {metrics.PositiveRecall}");
            //Console.WriteLine($"Accuracy: {metrics.F1Score}");

            ////Save the model to a file
            //mLContext.Model.Save(model, dataView.Schema, "sentiment_model.zip");

            string modelPath = "sentiment_model.zip";
            string testDataPath = "movieReviewsTesting.csv";
            var mlContext = new MLContext();
            ITransformer model;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                //Load the model from the file
                model = mlContext.Model.Load(stream, out var modelInputSchema);
            }

            //Load the test data from the file
            IDataView testData = mlContext.Data.LoadFromTextFile<TextData>(testDataPath, hasHeader: true, separatorChar: ',');

            //Apply the model to the test data
            var predictor = mlContext.Model.CreatePredictionEngine<TextData, SentimentPrediction>(model);

            //Get the predictions
            var testDataList = mlContext.Data.CreateEnumerable<TextData>(testData, reuseRowObject: false).ToList();
            foreach (var data in testDataList)
            {
                //Make a prediction
                var prediction = predictor.Predict(data);
                //Print the prediction
                Console.WriteLine($"Text: {data.text} | Prediction: {(prediction.IsPositiveSentiment ? "Positive" : "Negative")}");
            }
        }
    }
}

