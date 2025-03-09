using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Net.WebSockets;

namespace Recommendation
{
    public class MovieRating
    {
        [LoadColumn(0)]
        public float userId;

        [LoadColumn(1)]
        public float movieId;

        [LoadColumn(2)]
        public float Label;
    }

    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }

    public class Program
    {

        /*
        This method takes an MLContext object and an IDataView object as input. It performs data preprocessing by mapping the user ID and movie ID columns to key values. 
        This is done to make it easier for the recommendation model to process the data.  
         */
        public static IDataView PreProcessData(MLContext mLContext, IDataView dataView)
        {
            /*
              The user ID has remained the same, but the movie ID has been changed so that each movie ID is one after
              the other, sequentially, without any gaps.
              Furthermore, the ratings were converted from doubles into integers to make them easier to work with.
             */
            return mLContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userId", inputColumnName: "userId")
                    .Append(mLContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieId", inputColumnName: "movieId"))
                    .Fit(dataView).Transform(dataView);
        }

        /*
          This method takes an MLContext object, an IDataView object, and a file path as input. 
          It saves the preprocessed data to a file in CSV format.
         
         */
        public static void SaveData(MLContext mLContext, IDataView dataView, string dataPath)
        {
            using(var fileStream = new FileStream(dataPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mLContext.Data.SaveAsText(dataView, fileStream, separatorChar:',',headerRow:true, schema:false);
            }
        }

        /*
          This method takes an MLContext object as input and returns a tuple of IDataView objects. 
          It loads the preprocessed data from a CSV file and splits it into training and test data.
         */
        static (IDataView training, IDataView test) LoadData(MLContext mLContext)
        {
            var dataPath = "preprocessed_ratings.csv";
            IDataView fullData = mLContext.Data.LoadFromTextFile<MovieRating>(dataPath, hasHeader: true, separatorChar: ',');
            var trainTestData = mLContext.Data.TrainTestSplit(fullData, testFraction: 0.2);
            IDataView trainData = trainTestData.TrainSet;
            IDataView testData = trainTestData.TestSet;
            return (trainData, testData);
        }

        /*
         This method takes an IDataView object as input and prints a preview of the data. 
         It shows the key-value pairs for each row in the data.
         */
        public static void PrintDataPreview(IDataView dataView)
        {
            var preview = dataView.Preview();
            foreach (var row in preview.RowView)
            {
                foreach (var column in row.Values)
                {
                    Console.Write($"{column.Key}:{column.Value}\t");
                }
                Console.WriteLine();
            }
        }

        /*
          This method takes an MLContext object and an IDataView object as input. 
          It trains a recommendation model using the MatrixFactorizationTrainer and returns the trained model.
         */
        static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion
                                                .MapValueToKey(outputColumnName: "outputUserId", inputColumnName: "userId")
                                                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "outputMovieId", inputColumnName: "movieId"));

            /*
             An instance of the MatrixFactorizationTrainer.Options class is created. 
            This class contains various configuration options for the Matrix Factorization trainer.
             */

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "outputUserId",
                MatrixRowIndexColumnName = "outputMovieId",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            /*
             The estimator object is responsible for transforming the input data by mapping the user and movie IDs to key values.
             The estimator object will map the "userId" and "movieId" columns to key values. For example, it might map "userId" 1 to key value 0, "userId" 2 to key value 1, and so on. Similarly, it will map "movieId" 101 to key value 0, "movieId" 102 to key value 1, and so on.
             The trainerEstimator will then use the Matrix Factorization algorithm to train the model on the transformed data. 
             The model will learn the underlying patterns in the ratings data and make predictions for unseen user-movie combinations.
             */
            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));
            ITransformer model = trainerEstimator.Fit(trainingDataView);
            Console.WriteLine("Model successfully trained");
            return model;
        }

        /*
          This method takes an MLContext object, an IDataView object, and a trained model as input. 
          It evaluates the model's performance on the test dataset by calculating the RSquared and Root Mean Squared Error metrics.
         */
        static void EvaluateModel(MLContext mLContext, IDataView testDataView, ITransformer model)
        {
            var prediction = model.Transform(testDataView);
            var metrics = mLContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"RSquared: {metrics.RSquared}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
        }

        /*
          This method takes an MLContext object and a trained model as input. 
          It uses the model to make a single prediction for a user and a movie and prints the predicted rating.
         */

        static void UseModelForSinglePrediction(MLContext mLContext, ITransformer model)
        {
            var predictionEngine = mLContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            var testInput = new MovieRating { userId = 14, movieId = 433 };
            var movieRatingPrediction = predictionEngine.Predict(testInput);
            Console.WriteLine("Predicted rating for movie " + testInput.movieId + " is : " + Math.Round(movieRatingPrediction.Score, 1));
            string recommendation = Math.Round(movieRatingPrediction.Score, 1) > 3.5 ? 
                "Movie " + testInput.movieId + " is recommended for user " + testInput.userId :
                "Movie " + testInput.movieId + " is not recommended for user " + testInput.userId;
            Console.WriteLine(recommendation);
        }


        /*
        This method is the entry point of the program. 
        It initializes the MLContext object, loads the original data, preprocesses and saves the data, 
        loads the training and test datasets, prints a preview of the training data, 
        trains the model, evaluates the model, and makes a single prediction using the model.
         */
        public static void Main(string[] args)
        {
            var mLContext = new MLContext(seed: 0);

            var fullData = mLContext.Data.LoadFromTextFile<MovieRating>("ratings.csv", hasHeader: true, separatorChar: ',');

            var preprocessData = PreProcessData(mLContext, fullData);

            SaveData(mLContext, preprocessData, "preprocessed_ratings.csv");

            (IDataView trainingDataView, IDataView testDataView) data = LoadData(mLContext);

            PrintDataPreview(data.trainingDataView);
            ITransformer model = TrainModel(mLContext, data.trainingDataView);

            EvaluateModel(mLContext,data.testDataView, model);

            UseModelForSinglePrediction(mLContext, model);
        }
    }
}