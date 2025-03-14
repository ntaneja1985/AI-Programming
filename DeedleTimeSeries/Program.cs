using Deedle;
using ScottPlot;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

public class Program
{
    public static void Main()
    {
        //BasicDeedleManipulation();
        //LoadingAndManipulatingData();
        //BasicTimeSeriesOperations();
        //ResamplingAggregatingOperations();
        //TimeSeriesIndexingAndSlicing();
        //HandleMissingValuesInDeedle();
        //RollingWindowsAndMovingAverages();
        //TimeSeriesVisualization();
        //seasonalityAndTrendAnalysis();
        //StationarityTesting();
        MLNetIntegration();


    }

    public static void BasicDeedleManipulation()
    {
        var dates = new DateTime[] { new DateTime(2023,1,1)
                                    , new DateTime(2023,1,2)
                                    ,new DateTime(2023,1,3)};
        var values = new double[] { 10.5, 20.0, 30.2 };
        var series = new Series<DateTime, double>(dates, values);
        List<Series<DateTime, double>> seriesList = [series];
        var frame = Frame.FromColumns(seriesList);
    }

    public static void LoadingAndManipulatingData()
    {
        var dataFrame = Frame.ReadCsv("data (1).csv");
        Console.WriteLine(dataFrame);
        var timeSeries = dataFrame.GetColumn<double>("Value").SortByKey();
        //series [ 0 => 100; 1 => 150; 2 => 200; 3 => <missing>; 4 => 175;  ... ; 9 => 350]
        Console.WriteLine(timeSeries);
        var filteredSeries = timeSeries.Where(kvp => kvp.Value > 100);
        //series [ 1 => 150; 2 => 200; 4 => 175; 6 => 225; 7 => 250;  ... ; 9 => 350]
        Console.WriteLine(filteredSeries);

        //Provide an estimated value for missing elements in the timeseries
        var filledSeries = timeSeries.FillMissing(Direction.Forward);
        Console.WriteLine(filledSeries);
    }

    public static void BasicTimeSeriesOperations()
    {
        var dataFrame = Frame.ReadCsv("data (1).csv");
        Console.WriteLine(dataFrame);

        var specificDate = dataFrame.Rows[0]["Date"];
        Console.WriteLine(specificDate);

        var meanValue = dataFrame["Value"].Mean();
        Console.WriteLine(meanValue);

        var sumValue = dataFrame["Value"].Sum();
        Console.WriteLine(sumValue);

        dataFrame["Value"] = dataFrame["Value"].FillMissing(Direction.Forward);
        Console.WriteLine(dataFrame.Rows[3]);

        var cleanDataFrame = dataFrame.DropSparseRows();
        Console.WriteLine(cleanDataFrame.Rows[3]);
    }

    public static void ResamplingAggregatingOperations()
    {
        var dataFrame = Frame.ReadCsv("stock_data_1.csv");
        Console.WriteLine(dataFrame);
        var frameDate = dataFrame.IndexRows<DateTime>("Date").SortRowsByKey();
        var openingPriceByDay = frameDate.GetColumn<decimal>("Open");
        openingPriceByDay.Print();

        var openingPriceByWeek = openingPriceByDay
            .GroupBy(date => GetStartOfWeek(date.Key))
            .Select(g => new
            {
                WeekStart = g.Key,
                OpeningPrice = g.Value.Mean()
            });

        //27-05-2024 00:00:00 -> { WeekStart = 27-05-2024 00:00:00, OpeningPrice = 101.125 }
        //03 - 06 - 2024 00:00:00-> { WeekStart = 03 - 06 - 2024 00:00:00, OpeningPrice = 107.69285714285714 }
        //10 - 06 - 2024 00:00:00-> { WeekStart = 10 - 06 - 2024 00:00:00, OpeningPrice = 115.36428571428573 }
        //17 - 06 - 2024 00:00:00-> { WeekStart = 17 - 06 - 2024 00:00:00, OpeningPrice = 122.62142857142858 }
        //24 - 06 - 2024 00:00:00-> { WeekStart = 24 - 06 - 2024 00:00:00, OpeningPrice = 129.52857142857144 }
        //01 - 07 - 2024 00:00:00-> { WeekStart = 01 - 07 - 2024 00:00:00, OpeningPrice = 136.6142857142857 }
        //08 - 07 - 2024 00:00:00-> { WeekStart = 08 - 07 - 2024 00:00:00, OpeningPrice = 143.52857142857144 }
        //15 - 07 - 2024 00:00:00-> { WeekStart = 15 - 07 - 2024 00:00:00, OpeningPrice = 150.57142857142858 }
        openingPriceByWeek.Print();


    }

    private static DateTime GetStartOfWeek(DateTime date)
    {
        int daysToSubTract = (int)date.DayOfWeek - 1;
        if (daysToSubTract < 0) daysToSubTract += 7;
        return date.AddDays(-daysToSubTract).Date;
    }

    public static void TimeSeriesIndexingAndSlicing()
    {
        var dataFrame = Frame.ReadCsv("stock_data_1.csv");
        Console.WriteLine(dataFrame);
        var frameDate = dataFrame.IndexRows<DateTime>("Date").SortRowsByKey();
        var openingPriceByDay = frameDate.GetColumn<decimal>("Open");
        var firstDayOpeningPrice = openingPriceByDay.GetAt(0);
        Console.WriteLine(firstDayOpeningPrice);

        var openingPriceSpecificDay = openingPriceByDay.Get(new DateTime(2024, 6, 20));
        Console.WriteLine(openingPriceSpecificDay);

        var openingPricesBetweenTwoDays = openingPriceByDay.Between(new DateTime(2024, 6, 20), new DateTime(2024, 6, 30));
        Console.WriteLine(openingPricesBetweenTwoDays);

        var highValueData = dataFrame.Where(row => row.Value.GetAs<double>("Close") > 130);
        Console.WriteLine(highValueData);



    }

    public static void HandleMissingValuesInDeedle()
    {
        var dataFrame = Frame.ReadCsv("stock_data_with_missing_values.csv");
        Console.WriteLine(dataFrame);

        var fillZeroes = dataFrame.FillMissing(0.0);
        /*
        0  -> 01-06-2024 00:00:00 100.25 102.5  99.75  0
        1  -> 02-06-2024 00:00:00 102    104.75 0      104.5
        2  -> 03-06-2024 00:00:00 104.75 106.25 103.5  105.2
        3  -> 04-06-2024 00:00:00 105.4  107    104.8  0
        4  -> 05-06-2024 00:00:00 106.8  108.25 105.9  107.9
        5  -> 06-06-2024 00:00:00 107.5  109    106.75 108.4
        6  -> 07-06-2024 00:00:00 0      110.5  107.8  109.7
        7  -> 08-06-2024 00:00:00 109.8  111.25 109    110.9
        8  -> 09-06-2024 00:00:00 111    112.75 0      112.1
        9  -> 10-06-2024 00:00:00 0      0      0      113.5
        10 -> 11-06-2024 00:00:00 113.6  115.25 112.5  114.9
        11 -> 12-06-2024 00:00:00 114.5  0      113.8  115.2
         */
        //fillZeroes.Print();

        var meanValueForColumn = dataFrame.GetColumn<double>("Close").Mean();
        dataFrame = dataFrame.FillMissing(meanValueForColumn);
        dataFrame.Print();

        //Linear Interpolation
        var closeSeries = dataFrame.GetColumn<double>("Close");
        var interpolatedLinear = closeSeries.Interpolate(
            closeSeries.Keys,
            (key, prev, next) =>
            {
                if (prev.HasValue && next.HasValue)
                {
                    return (prev.Value.Value + next.Value.Value) / 2;
                }
                return prev.HasValue ? prev.Value.Value : next.Value.Value;
            }
            );

        interpolatedLinear.Print();

    }

    public static void RollingWindowsAndMovingAverages()
    {
        var dataFrame = Frame.ReadCsv("rollingwindow.csv");
        Console.WriteLine(dataFrame);

        var ThreedayRollingMean = dataFrame.GetColumn<double>("Value")
                            .Window(3)
                            .SelectValues(win => win.Mean());

        ThreedayRollingMean.Print();

        var ThreedayRollingSum = dataFrame.GetColumn<double>("Value")
                           .Window(3)
                           .SelectValues(win => win.Sum());

        ThreedayRollingSum.Print();

        var ThreedayRollingStandardDeviation = dataFrame.GetColumn<double>("Value")
                           .Window(3)
                           .SelectValues(win => win.StdDev());

        ThreedayRollingStandardDeviation.Print();
    }

    public static void TimeSeriesVisualization()
    {
        var dataFrame = Frame.ReadCsv("timseriesvisualization.csv");
        Console.WriteLine(dataFrame);
        dataFrame.IndexRows<DateTime>("Date");
        Plot plot = new Plot();
        plot.Title("Time Series Scatter Plot");
        plot.XLabel("Date");
        plot.YLabel("Value");
        var dates = dataFrame.RowKeys.ToArray();
        var values = dataFrame.GetColumn<double>("Value").Values.ToArray();
        //plot.Add.Scatter(dates, values);
        //plot.SavePng("vis1.png", 500, 500);

        for (int i = 0; i < dates.Length; i++)
        {
            var bar = new Bar
            {
                Position = i,
                Value = values[i],
                Size = 0.8,
                FillColor = Colors.Blue,
                LineColor = Colors.Black
            };
            plot.Add.Bar(bar);
        }

        plot.SavePng("vis2.png", 500, 500);
    }

    public static void seasonalityAndTrendAnalysis()
    {
        var dataFrame = Frame.ReadCsv("seasonality.csv");
        Console.WriteLine(dataFrame);

        var movingAverage = dataFrame.GetColumn<double>("Value")
                            .Window(12).SelectValues(win => win.Mean());
        dataFrame.AddColumn("Trend", movingAverage);

        var seasonalIndex = dataFrame["Value"] / dataFrame["Trend"];
        dataFrame.AddColumn("Seasonal Index", seasonalIndex);

        var residual = dataFrame["Value"] - dataFrame["Trend"] * dataFrame["SeasonalIndex"];
        dataFrame.AddColumn("Residual", residual);
        dataFrame.Print();

        Plot plot = new Plot();
        plot.Title("Trend");
        plot.XLabel("Date");
        plot.YLabel("Trend Component");
        dataFrame.IndexRows<DateTime>("Date");
        var dates = dataFrame.RowKeys.ToArray();
        var trendValues = dataFrame.GetColumn<double>("Trend").Values.ToArray();
        plot.Add.Scatter(dates, trendValues);
        plot.SavePng("trends.png", 500, 500);
    }

    public static void StationarityTesting()
    {
        var dataFrame = Frame.ReadCsv("seasonality.csv");
        Console.WriteLine(dataFrame);
        var values = dataFrame["Value"].Values.ToArray();
        var isStationary = IsStationary(values);
        Console.WriteLine(isStationary);
    }

    static bool IsStationary(double[] values, double significanceLevel = 0.05)
    {
        double mean = values.Average();
        double variance = values.Select(v => (v-mean)*(v-mean)).Sum()/(values.Length-1);
        double adfStat = variance / (Math.Sqrt(variance) * values.Length);
        double criticalValue = -3.5;
        return adfStat < criticalValue;
    }

    public static void MLNetIntegration()
    {
        var dataFrame = Frame.ReadCsv("finalDataTimeSeries.csv");
        dataFrame = dataFrame.SortRowsBy<int, string,string, DateTime>("Date",dateString => DateTime.Parse(dateString));

        var timeSeriesList = new List<TimeSeriesData>();
        var timeSeriesData = dataFrame.Rows.Select(row => new TimeSeriesData
        {
            Date = row.Value.GetAs<DateTime>("Date"),
            Value = row.Value.GetAs<float>("Value")
        });

        foreach (var item in timeSeriesData.GetAllValues())
        {
            timeSeriesList.Add(item.Value);
        }

        var mlContext = new MLContext();
        IDataView dataView = mlContext.Data.LoadFromEnumerable(timeSeriesList);
        var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
            "ForecastedValues", "Value", windowSize: 12, seriesLength: 24, trainSize: 36, horizon: 6
            );

        var model = forecastingPipeline.Fit(dataView);
        var forecastEngine = model.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
        var forecast = forecastEngine.Predict();

        Console.WriteLine("Forecasted Values: ");
        foreach(var value in forecast.ForecastedValues)
        {
            Console.WriteLine(value.ToString());
        }

        var predictions = model.Transform(dataView);
        var predictionResults = mlContext.Data.CreateEnumerable<ModelOutput>(predictions, reuseRowObject: false).ToList();
        foreach(var result in predictionResults)
        {
            Console.WriteLine(string.Join(",",result.ForecastedValues));
        }
    }
}

public class TimeSeriesData
{
    public DateTime Date { get; set; }
    public float Value { get; set; }
}

public class ModelInput
{
    [LoadColumn(0)]
    public DateTime Date { get; set; }

    [LoadColumn(1)]
    public float Value { get; set; }
}

public class ModelOutput
{
    [VectorType(6)]
    public float[] ForecastedValues { get; set; }
}