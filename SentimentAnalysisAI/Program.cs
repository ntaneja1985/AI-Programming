using Microsoft.ML;
using Microsoft.ML.Data;

public class SentimentData
{
    [LoadColumn(0)]
    public string review { get; set; }

    [LoadColumn(1)]
    [ColumnName("Label")]
    public bool sentiment { get; set; }
}

public class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();
        string dataPath = "movieReviews.csv";
        string text = File.ReadAllText(dataPath);

        //Remove single quotes from the csv file
        //Replace the words positive and negative with true and false
        using (StreamReader reader = new StreamReader(dataPath))
        {
            text = text.Replace("\'", "");
            text = text.Replace("positive", "true");
            text = text.Replace("negative", "false");
        }

        File.WriteAllText(dataPath, text);

        IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: true,allowQuoting:true, separatorChar: ',');

        Console.WriteLine("Data loaded successfully");
        Console.WriteLine();
        var preview = dataView.Preview(maxRows: 5);
        foreach (var row in preview.RowView)
        {
            foreach (var column in row.Values)
            {
                Console.WriteLine($"{column.Key}: {column.Value}");
            }
        }



        var trainTestSplit  = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

        var trainData = trainTestSplit.TrainSet;

        /*
         We create a pipeline that is responsible for transforming the text data in the "review" column into numerical features. 
          It uses the FeaturizeText method from the Text transforms in the MLContext to convert the text into a numerical representation that can be used by the machine learning algorithm. 
          The transformed features are stored in a new column called "Features".
         After this we append a binary classification trainer to the previous transformation. 
          It uses the SdcaLogisticRegression trainer from the BinaryClassification trainers in the MLContext. 
          The trainer is responsible for training a logistic regression model to predict the sentiment label based on the transformed features. 
          The "Label" column is used as the target label, and the "Features" column is used as the input features for training the model.
         */
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "review")
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));

        //Fit the pipeline to the training data
        var model = pipeline.Fit(trainData);

        var testData = trainTestSplit.TestSet;

        //Make predictions on the test data
        var predictions = model.Transform(testData);

        //Evaluate the model
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

        //How often the AI gets the sentiment(positive or negative) correct
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        //AUC stands for Area under ROC Curve. It means how well the AI can tell the difference between
        //positive and negative sentiments
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
        //F1 Score is the balance between how many positive reviews the AI correctly finds(recall) and how many of the reviews it says are positive that actually are positive(precision)
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
        //Log loss also known as cross-entropy loss measures how confident the AI is in its predictions and
        //how wrong it is when it makes mistakes
        Console.WriteLine($"Log Loss: {metrics.LogLoss:F2}");
    }
}