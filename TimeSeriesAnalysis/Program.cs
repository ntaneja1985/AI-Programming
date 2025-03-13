using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TimeSeries;
using System.Globalization;


public class TrafficData
{
    [LoadColumn(0)]
    public DateTime Date;

    [LoadColumn(1)]
    public float Traffic;
}

public class Prediction
{
    public float[] PredictedTraffic { get; set; }
    public float[] LowerBoundTraffic { get; set; }
    public float[] UpperBoundTraffic { get; set; }
}
public class Program
{

    // Forecast the traffic values for the next 'horizon' time steps using the forecasting model.
    static void Forecast(MLContext mlContext, IDataView testData,int horizon, TimeSeriesPredictionEngine<TrafficData,Prediction> forecaster)
    {
        var forecast = forecaster.Predict(horizon: horizon);
        var testTrafficData = mlContext.Data.CreateEnumerable<TrafficData>(testData, reuseRowObject: false).ToList();
        for (int i = 0; i < horizon && i< forecast.PredictedTraffic.Length; i++)
        {
           string date = testTrafficData[i].Date.ToShortDateString();
           float actualTraffic = i < testTrafficData.Count ?  testTrafficData[i].Traffic : 0;
            float lowerEstimate = Math.Max(0,forecast.LowerBoundTraffic[i]);
            float estimate = forecast.PredictedTraffic[i];
            float upperEstimate = forecast.UpperBoundTraffic[i];
            Console.WriteLine($"Date: {date}, Actual Traffic: {actualTraffic}, Forecast Traffic: {estimate}, Lower Estimate: {lowerEstimate}, Upper Estimate: {upperEstimate}");
        }
    }

    // Evaluate the forecasting model using the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics.
    static void EvaluateMetrics(IDataView testData, IDataView predictions, MLContext mLContext)
    {
        IEnumerable<float> actual = mLContext.Data.CreateEnumerable<TrafficData>(testData,true)
            .Select(observed => observed.Traffic);
        IEnumerable<float> forecast = mLContext.Data.CreateEnumerable<Prediction>(predictions, true)
            .Select(prediction => prediction.PredictedTraffic[0]);
        
        var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);
        var MAE = metrics.Average(Math.Abs);
        var RMSE = Math.Sqrt(metrics.Average(error =>Math.Pow(error,2)));
        Console.WriteLine("Evaluate Metrics");
        Console.WriteLine($"Mean Absolute Error: {MAE}");
        Console.WriteLine($"Root Mean Squared Error: {RMSE}");
    }
    public static void Main(string[] args)
    {
        string dataPath = "data_timeseries.csv";
        var mlContext = new MLContext();
        // Load data
        var dataView = mlContext.Data.LoadFromTextFile<TrafficData>(dataPath, hasHeader: true, separatorChar: ',');

        // It replaces missing values in the "Traffic" column of the data with the mean value of the available data. It is a preprocessing step that ensures the data is complete and ready for further analysis or modeling.
        var filledDataView = mlContext.Transforms.ReplaceMissingValues(
            outputColumnName: "Traffic",
            replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
            .Fit(dataView)
            .Transform(dataView);

        var trainTestData = mlContext.Data.TrainTestSplit(filledDataView, testFraction: 0.2);
        var trainData = trainTestData.TrainSet;
        var testData = trainTestData.TestSet;

        //Create a forecasting pipeline using the Singular Spectrum Analysis (SSA) algorithm in the ML.NET library.
        //The SSA algorithm is used for time series forecasting, which involves predicting future values based on historical data.
        var pipeline = mlContext.Forecasting.ForecastBySsa(
            //Specifies the name of the column that will store the predicted traffic values.
            outputColumnName: "PredictedTraffic",
            //Specifies the name of the column that contains the input traffic values used for forecasting.
            inputColumnName: "Traffic",
            //Specifies the size of the sliding window used in the SSA algorithm.
            //This window size determines the number of historical data points used to make predictions.
            windowSize: 14,
            //Specifies the length of the time series.
            //This value represents the total number of data points in the time series.
            seriesLength: 100,
            //Specifies the size of the training set, which is the percentage of the time series used for training the forecasting model.In this case, 80 % of the data will be used for training.
            trainSize: 80,
            //Specifies the number of future time steps to forecast.In this case, the model will predict the traffic values for the next 7 time steps.
            horizon: 7,
            //Specifies the confidence level for the prediction intervals.
            //The confidence level of 0.95 means that the predicted traffic values will fall within the 95% confidence interval.
            confidenceLevel: 0.95f,
            //Specifies the name of the column that will store the lower bound of the confidence interval.
            confidenceLowerBoundColumn: "LowerBoundTraffic",
            //Specifies the name of the column that will store the upper bound of the confidence interval.
            confidenceUpperBoundColumn: "UpperBoundTraffic");

        var model = pipeline.Fit(trainData);
        var predictions = model.Transform(testData);
        //EvaluateMetrics(testData, predictions, mlContext);

        //Create a time series engine using the forecasting model.
        var forecastingEngine = model.CreateTimeSeriesEngine<TrafficData, Prediction>(mlContext);
        Forecast(mlContext, testData, 7, forecastingEngine);
    }
}
