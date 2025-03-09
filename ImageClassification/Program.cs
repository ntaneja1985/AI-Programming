using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using static Microsoft.ML.DataOperationsCatalog;

public class ImageData
{
    [LoadColumn(0)]
    public string? ImagePath { get; set; }

    [LoadColumn(1)]
    public string Label { get; set; }
}

class InputData
{
    public byte[] Image { get; set; }
    public uint LabelKey { get; set; }

    public string ImagePath { get; set; }
    public string Label { get; set; }
}

class  Output
{
    public string ImagePath { get; set; }
    public string Label { get; set; }
    public string PredictedLabel { get; set; }
}

public class Program
{
    static string dataFolder = "C:\\GithubCode\\AI-Programming\\ImageClassification\\Data";

    private static IEnumerable<ImageData> LoadImagesFromDirectory(string folder)
    {
        var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);
        foreach(var file in files)
        {
            if ((Path.GetExtension(file) != ".jpg") &&
                (Path.GetExtension(file) != ".png") &&
                (Path.GetExtension(file) != ".jpeg"))
                continue;

            string label = Path.GetFileNameWithoutExtension(file).Trim();
            label = label.Substring(0, label.Length - 1);
            yield return new ImageData()
            {
                ImagePath = file,
                Label = label
            };
        }
    }

    public static void PrintDataView(IDataView dataView)
    {
        var preview = dataView.Preview();
        foreach (var row in preview.RowView)
        {
            foreach (var kvp in row.Values)
            {
                Console.Write($"{kvp.Key}:{kvp.Value} ");
            }
            Console.WriteLine();
        }
    }

    private static void OutputPrediction(Output prediction)
    {

        string imageName = Path.GetFileName(prediction.ImagePath);
        Console.WriteLine($"Image: {imageName} | Actual Label: {prediction.Label} | Predicted Label: {prediction.PredictedLabel}");
    }

    private static void ClassifyMultiple(MLContext mLContext, IDataView data, ITransformer trainedModel)
    {
        IDataView predictedData = trainedModel.Transform(data);

        var predictions = mLContext.Data.CreateEnumerable<Output>(predictedData, reuseRowObject: false).ToList();

        Console.WriteLine("AI Predictions: ");
        foreach (var prediction in predictions.Take(4))
        {
            OutputPrediction(prediction);
        }
    }

        public static void Main()
    {
        MLContext mLContext = new MLContext();
        IEnumerable<ImageData> images = LoadImagesFromDirectory(dataFolder);
        IDataView imageData = mLContext.Data.LoadFromEnumerable(images);

        //Shuffle the data 
        IDataView shuffledData = mLContext.Data.ShuffleRows(imageData);

        //PrintDataView(shuffledData);

        /*
         In this part, a preprocessing pipeline is created using the mLContext.Transforms API. The pipeline consists of two transformations:
        •	MapValueToKey: This transformation maps the string labels in the "Label" column to numeric keys in the "LabelKey" column. 
        This is necessary because machine learning algorithms typically work with numeric labels. 
        For example, if you have labels like "cat", "dog", and "bird", they will be mapped to numeric keys like 0, 1, and 2.
        •	LoadRawImageBytes: This transformation loads the raw image bytes from the specified image folder and stores them in the "Image" column. 
        It takes the "ImagePath" column as input, which contains the file paths of the images. The loaded image bytes can be used as input for image classification models.
        Example: Suppose you have a dataset with the following rows: | ImagePath       | Label  | |-----------------|--------| | image1.jpg      | cat    | | image2.jpg      | dog    | | image3.jpg      | bird   |
        After applying the preprocessing pipeline, the resulting dataset will have the following columns: | ImagePath       | Label  | LabelKey | Image (raw image bytes) | |-----------------|--------|----------|------------------------| | image1.jpg      | cat    | 0        | [raw image bytes]      | | image2.jpg      | dog    | 1        | [raw image bytes]      | | image3.jpg      | bird   | 2        | [raw image bytes]      |
         */
        var preprocessingPipeline = mLContext.Transforms.Conversion
            .MapValueToKey(inputColumnName:"Label", outputColumnName:"LabelKey")
            .Append(mLContext.Transforms.LoadRawImageBytes(
                outputColumnName: "Image",
                imageFolder: dataFolder,
                inputColumnName: "ImagePath"));




        IDataView preProcessedData = preprocessingPipeline.Fit(shuffledData).Transform(shuffledData);

        /*
         In this part, the preprocessed data is split into a training set and a test set using the TrainTestSplit method. 
        The testFraction parameter specifies the fraction of data to be used for testing (in this case, 40%). The remaining data is used for training.
         Example: Suppose the preprocessed data contains 100 rows. After the train-test split, the training set will contain 60 rows (60% of the data) and the test set will contain 40 rows (40% of the data).
         */
        TrainTestData trainTestData = mLContext.Data.TrainTestSplit(preProcessedData, testFraction: 0.4);
        IDataView trainSet = trainTestData.TrainSet;
        IDataView testSet = trainTestData.TestSet;


        /*
        In this part, the options for the image classification trainer are set. These options define how the model will be trained. Here are the key options:
        •	FeatureColumnName: Specifies the name of the column that contains the input image data (raw image bytes).
        •	LabelColumnName: Specifies the name of the column that contains the numeric label keys.
        •	ValidationSet: Specifies the test set to be used for validation during training.
        •	Arch: Specifies the architecture of the image classification model. In this case, the ResNet v2 101 architecture is used.
        •	MetricsCallback: Specifies a callback function that will be called during training to print the metrics (e.g., accuracy, loss) to the console.
        •	TestOnTrainSet: Specifies whether to evaluate the model on the training set during training. In this case, it is set to false.
        •	ReuseTrainSetBottleneckCachedValues and ReuseValidationSetBottleneckCachedValues: These options control whether to reuse the cached bottleneck values during training. Bottleneck values are intermediate representations of the images used to speed up training.
         
         */
        var classifierOptions = new ImageClassificationTrainer.Options()
        {
            FeatureColumnName = "Image",
            LabelColumnName = "LabelKey",
            ValidationSet = testSet,
            Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
            MetricsCallback = (metrics) => Console.WriteLine(metrics),
            TestOnTrainSet = false,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true,
            //WorkspacePath = "C:\\GithubCode\\AI-Programming\\ImageClassification\\Data"
        };


        /*
        In this part, the training pipeline is created using the mLContext.MulticlassClassification.Trainers.ImageClassification method. 
        The pipeline consists of the image classification trainer followed by a transformation to map the predicted label keys back to their original string labels.
        Example: The training pipeline takes the preprocessed training set as input and trains an image classification model. 
        The trained model can then be used to make predictions on new images.
         */
        var trainingPipeline = mLContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                                .Append(mLContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // Train the model
        // During the training process, the model learns patterns and relationships between the input images and their corresponding labels.
        // The trained model can then be used to make predictions on new, unseen images.
        ITransformer trainedModel = trainingPipeline.Fit(trainSet);

        //Responsible for using the trained model to make predictions on the test data and printing the results.
        ClassifyMultiple(mLContext, testSet, trainedModel);

    }
}