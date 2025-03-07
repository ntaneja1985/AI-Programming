using Microsoft.ML;
using Microsoft.ML.Data;


static void EvaluateMetrics(string modelName, BinaryClassificationMetrics metrics)
{
    Console.WriteLine($"{modelName} - Accuracy:{metrics.Accuracy:0.##}");
    Console.WriteLine($"{modelName} - AUC:{metrics.AreaUnderRocCurve:0.##}");
}

var context = new MLContext();
var data = context.Data.LoadFromTextFile<DataPoint>("data.csv", separatorChar: ',', hasHeader:true);

// Split the data into training and test sets
var trainTestSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

// Train the model
// Define the pipeline
// Concatenate the features into a single column
// Append the logistic regression trainer
// The label column is the "Label" column
// The maximum number of iterations is 100
// context.Transforms.Concatenate("Features", "Feature1", "Feature2") is a transformation step that concatenates multiple input features into a single column.
// In this example, we are concatenating two features, "Feature1" and "Feature2", into a new column called "Features".
// This transformation is useful when you want to combine multiple features into a single input for the model.
// Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName:"Label", maximumNumberOfIterations: 100)) is the trainer step that appends a logistic regression trainer to the pipeline. The trainer is responsible for training the model using the transformed data. In this example, we are using the SdcaLogisticRegression trainer, which is a type of logistic regression algorithm.
// We specify the label column name as "Label" and set the maximum number of iterations to 100.
var logisticRegressionPipeline = context.Transforms.Concatenate("Features", "Feature1", "Feature2")
    .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName:"Label", maximumNumberOfIterations: 100));


var fastTreePipeline = context.Transforms.Concatenate("Features", "Feature1", "Feature2")
    .Append(context.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", numberOfLeaves: 50, numberOfTrees:100));

Console.WriteLine("Training Logistic Regression model...");
var logisticRegressionModel = logisticRegressionPipeline.Fit(trainTestSplit.TrainSet);

Console.WriteLine("Training FastTree model...");
var fastTreeModel = fastTreePipeline.Fit(trainTestSplit.TrainSet);

// Evaluate the models
Console.WriteLine("Evaluating the Logistic Regression Model...");
var logisticRegressionPredictions = logisticRegressionModel.Transform(trainTestSplit.TestSet);
var logisticRegressionMetrics = context.BinaryClassification.Evaluate(logisticRegressionPredictions, "Label");
EvaluateMetrics("Logistic Regression", logisticRegressionMetrics);

Console.WriteLine("Evaluating the FastTree Model...");
var fastTreePredictions = fastTreeModel.Transform(trainTestSplit.TestSet);
var fastTreeMetrics = context.BinaryClassification.Evaluate(fastTreePredictions, "Label");
EvaluateMetrics("FastTree", fastTreeMetrics);

if(logisticRegressionMetrics.Accuracy > fastTreeMetrics.Accuracy)
{
    Console.WriteLine("Logistic Regression Model is the better model");
} else if(logisticRegressionMetrics.Accuracy < fastTreeMetrics.Accuracy)
{
    Console.WriteLine("FastTree Model is the better model");
}
else
{
    Console.WriteLine("Both models are equally good");
}

public class DataPoint
{
    [LoadColumn(0)]

    public float Feature1 { get; set; }
    [LoadColumn(1)]
    public float Feature2 { get; set; }

    [LoadColumn(2)]
    public bool Label { get; set; }
}

public class Prediction
{
    [ColumnName("Score")]
    public float Score { get; set; }

    [ColumnName("Probability")]
    public float Probability { get; set; }
}

