using System.Data;
using Accord.IO;
using Accord.Statistics;
using Accord.Statistics.Filters;
using ScottPlot;
using Accord.Statistics.Testing;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.MachineLearning;
using Accord.Statistics.Models.Regression.Linear;
using Accord.MachineLearning;
using Accord.Math;
using Accord.Statistics;
using Accord.Statistics.Analysis;
using Accord.MachineLearning.Clustering;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Neuro;
using Accord.Neuro.Learning;

public class Program
{
    public static void Main()
    {
        //LoadingData();
        //EDA();
        //StatisticalAnalysis();
        //HypothesisTests();
        //ClassificationAlgos();
        //RegressionTechniques();
        //ClusteringTechniques();
        //DimensionalityReductionTechniques();
        //EnsembleLearning();
        //SupportVectorMachines();
        NeuralNetworks();
    }

    private static void NeuralNetworks()
    {
        //provide number of input neurons, number of hidden neurons and number of output neurons
        ActivationNetwork network = new ActivationNetwork(new SigmoidFunction(),2,2,1);
        BackPropagationLearning teacher = new BackPropagationLearning(network)
        {
            LearningRate = 0.1
        };
        double[][] inputs =
        {
            [0,0],
            [0,1],
            [1,0],
            [0,1]
        };
        double[][] outputs = {
        [0] ,
        [1] ,
        [1],
        [0]
        };

        //Train for 1000 episodes or epochs
        for (int i = 0; i < 1000; i++)
        {
            double error = teacher.RunEpoch(inputs, outputs);
            if(i % 100 == 0)
            {
                Console.WriteLine($"Epoch {i}, Error: {error}");
            }
        }

        //provide a sample input
        double[] result = network.Compute([0, 1]);
        Console.WriteLine($"Result for input [0,1]: {result[0]}");
    }

    private static void SupportVectorMachines()
    {
        double[][] inputs =
        {
            [0,0],
            [0,1],
            [1,0],
            [1,1],
        };

        int[] xor = { 0, 1, 1, 0 };
        var learn = new SequentialMinimalOptimization<Gaussian>()
        {
            UseComplexityHeuristic = true,
            UseKernelEstimation = true,
        };

        SupportVectorMachine<Gaussian> svm = learn.Learn(inputs, xor);
        bool[] predictions = svm.Decide(inputs);
        for (int i = 0; i < inputs.Length; i++)
        {
            int prediction = predictions[i] ? 1 : 0;
            Console.WriteLine($"Input: ({inputs[i][0]},{inputs[i][1]}) - Prediction: {prediction}");
        }
    }

    private static void EnsembleLearning()
    {
        string[] lines = File.ReadAllLines("ensembleLearning.csv");
        double[][] data = new double[lines.Length - 1][];
        int[] labels = new int[lines.Length - 1];
        for (int i = 0; i < lines.Length; i++)
        {
            string[] parts = lines[i].Split(',');
            data[i - 1] = new double[]
            {
                double.Parse(parts[1]), //Math Score
                double.Parse(parts[2]), //English Score
                double.Parse(parts[3]), //Science Score
                double.Parse(parts[4]) //History Score
            };
            labels[i-1] = int.Parse(parts[4]);
        }

        DecisionVariable[] attributes =
        {
            new DecisionVariable("MathScore",DecisionVariableKind.Continuous),
            new DecisionVariable("EnglishScore",DecisionVariableKind.Continuous),
            new DecisionVariable("ScienceScore",DecisionVariableKind.Continuous),
            new DecisionVariable("HistoryScore",DecisionVariableKind.Continuous),
        };

        var teacher = new RandomForestLearning()
        {
            NumberOfTrees = 100
        };

        var forest = teacher.Learn(data,labels);
        int[] predictions = forest.Decide(data);

        double accuracy = new GeneralConfusionMatrix(predictions, labels).Accuracy;
        Console.WriteLine($"Accuracy: {accuracy * 100:0.00}");

    }

    private static void DimensionalityReductionTechniques()
    {
        string[] lines = File.ReadAllLines("clustering.csv");
        double[][] data = new double[lines.Length - 1][];
        for (int i = 0; i < lines.Length; i++)
        {
            string[] parts = lines[i].Split(',');
            data[i - 1] = new double[]
            {
                double.Parse(parts[1]), //Math Score
                double.Parse(parts[2]), //English Score
                double.Parse(parts[3]), //Science Score
                double.Parse(parts[4]) //History Score
            };
        }

        var pca = new PrincipalComponentAnalysis()
        {
            Method = PrincipalComponentMethod.Center,
            Whiten = false
        };

        pca.Learn(data);
        double[][] pcaResult = pca.Transform(data);
        Console.WriteLine("PCA Results:");
        for(int i = 0; i < pcaResult.Length ; i++)
        {
            Console.WriteLine($"Data point {i + 1}: [{string.Join(", ", pcaResult[i])}]");
        }

        var tsne = new TSNE()
        {
            NumberOfOutputs = 2,
            Perplexity = 2,
            Theta = 0.5
        };
        double[][] tsneResult = tsne.Transform(data);
        Console.WriteLine("TSNE Results:");
        for (int i = 0; i < tsneResult.Length; i++)
        {
            Console.WriteLine($"Data point {i + 1}: [{string.Join(", ", tsneResult[i])}]");
        }

    }

    private static void ClusteringTechniques()
    {
        string[] lines = File.ReadAllLines("clustering.csv");
        double[][] data = new double[lines.Length - 1][];
        for (int i = 0; i < lines.Length; i++)
        {
            string[] parts = lines[i].Split(',');
            data[i - 1] = new double[]
            {
                double.Parse(parts[1]), //Math Score
                double.Parse(parts[2]), //English Score
                double.Parse(parts[3]), //Science Score
                double.Parse(parts[4]) //History Score
            };

        }
        int k = 3;
        KMeans kmeans = new KMeans(k);
        KMeansClusterCollection clusters = kmeans.Learn(data);
        int[] labels = clusters.Decide(data);
        Console.WriteLine("K-Means Clustering Results");
        for (int i = 0; i < labels.Length; i++)
        {
            Console.WriteLine($"Data point {i + 1} is in cluster {labels[i] + 1}");
        }
        
    }

    private static void RegressionTechniques()
    {
        double[][] inputs =
       {
            [1],
            [2],
            [3],
            [4]
        };
        double[] outputs = { 2, 3, 5, 7 };

        var ols = new OrdinaryLeastSquares();
        var regression = ols.Learn(inputs, outputs);
        double[] newInput = { 5 };
        double predicted = regression.Transform(newInput);
        Console.WriteLine(predicted);

        double[] inputsPloy =
       {1,2,3,4};
        double[] outputsPoly = { 2, 3, 5, 7 };
        var polly = new PolynomialRegression();
        polly.Regress(inputsPloy, outputsPoly);
        double newInputPolly = 5;
        double predictedPolly = polly.Transform(newInputPolly);
        Console.WriteLine(predictedPolly);
    }

    private static void ClassificationAlgos()
    {
        double[][] inputs =
        {
            [1,1],
            [1,0],
            [0,1],
            [0,0]
        };
        int[] outputs = { 1, 0, 0, 0 };
        var decisionVariables = DecisionVariable.FromData(inputs);
        var decisionTree = new DecisionTree(decisionVariables, 2);
        var teacher = new C45Learning(decisionTree);
        DecisionTree tree = teacher.Learn(inputs, outputs);
        int[] newInputs = { 1, 1 };
        int predicted = tree.Decide(newInputs);
        Console.WriteLine($"PredictedLabel: {predicted}");

        var knn = new KNearestNeighbors(k: 2);
        knn.Learn(inputs, outputs);
        double[] newInput = {1,1};
        int predictedVal = knn.Decide(newInput);
        Console.WriteLine($"Predicted Label: {predictedVal}");
    }

    public static void LoadingData()
    {
        var csvReader = new CsvReader("accorddata.csv", hasHeaders: true);
        DataTable dataTable = csvReader.ToTable();
        Normalization normalization = new Normalization(dataTable);
        DataTable result = normalization.Apply(dataTable);
        DisplayDataTable(result);

    }

    private static void DisplayDataTable(DataTable table)
    {
        foreach(DataColumn column in table.Columns)
        {
            Console.Write($"{column.ColumnName}\t");
        }
        Console.WriteLine();

        foreach(DataRow row in table.Rows)
        {
            foreach(var item in row.ItemArray)
            {
                Console.Write($"{item}\t");
            }

            Console.WriteLine();
        }
    }

    public static void EDA()
    {
        var csvReader = new CsvReader("eda.csv", hasHeaders: true);
        DataTable dataTable = csvReader.ToTable();

        //Descriptive Statistics
        foreach(DataColumn column in dataTable.Columns)
        {
            var values = dataTable.AsEnumerable().Select(row => Convert.ToDouble(row[column])).ToArray();
            double mean = values.Mean();
            double median = values.Median();
            double stdDev = values.StandardDeviation();

            double[] xValues = dataTable.AsEnumerable().Select(row => Convert.ToDouble(row["Feature"])).ToArray();
            double[] yValues = dataTable.AsEnumerable().Select(row => Convert.ToDouble(row["Target"])).ToArray();
            Plot plot = new Plot();
            plot.Title("Scatter Plot");
            plot.XLabel("Feature X Values");
            plot.YLabel("Feature Y Values");
            plot.Add.Scatter(xValues, yValues);
            plot.SavePng("demo.png", 500, 500);
        }
    }

    public static void StatisticalAnalysis()
    {
        var csvReader = new CsvReader("statistical.csv", hasHeaders: true);
        DataTable dataTable = csvReader.ToTable();

        foreach( DataColumn column in dataTable.Columns)
        {
            //Descriptive Statistics
            var values = dataTable.AsEnumerable().Select(row => Convert.ToDouble(row[column])).ToArray();
            double mean = values.Mean();
            double median = values.Median();
            double stdDev = values.StandardDeviation();
            double variance = values.Variance();
            double min = values.Min();
            double max = values.Max();
        }
    }

    public static void HypothesisTests()
    {
        double[] group1 = { 85, 89, 92, 88, 90, 91, 87 };
        double[] group2 = { 78, 74, 77, 76, 80, 79, 75 };

        //Tests where mean of 2 samples is different
        var ttest = new TwoSampleTTest(group1, group2, assumeEqualVariances: true);

        Console.WriteLine(ttest.Statistic);
        Console.WriteLine(ttest.PValue);
        Console.WriteLine(ttest.Significant);

        var csvReader = new CsvReader("student_scores.csv", hasHeaders: true);
        DataTable dataTable = csvReader.ToTable();
        var methodA = dataTable.AsEnumerable().Where(row => row["Method"].ToString() == "A")
                        .Select(row => Convert.ToDouble(row["Score"])).ToArray();

        var methodB = dataTable.AsEnumerable().Where(row => row["Method"].ToString() == "B")
                        .Select(row => Convert.ToDouble(row["Score"])).ToArray();

        double meanA = methodA.Mean();
        double meanB = methodB.Mean();

        double stdDevA = methodA.StandardDeviation();
        double stdDevB = methodB.StandardDeviation();

        var ttest2 = new TwoSampleTTest(methodA, methodB, assumeEqualVariances: true);
        Console.WriteLine(ttest2.Statistic);
        Console.WriteLine(ttest2.PValue);
        Console.WriteLine(ttest2.Significant);

    }
}