public class Program
{
    public static void Main()
    {
        List<double> data = new List<double>() { 2,4,4,5,5,7,9};
        // Calculate the mean
        // The mean is the average of a set of numbers calculated by dividing the sum of all the numbers by the
        //count of numbers.
        double mean = data.Average();
        // Calculate the median
        // The median is the middle number in a sorted, ascending or descending, list of numbers and can be more
        double median = data.OrderBy(x => x).Skip(data.Count / 2).First();
        // Calculate the mode
        // The mode is the number that appears most frequently in a data set.
        double mode = data.GroupBy(n=>n).OrderByDescending(g => g.Count()).First().Key;
        // Calculate the standard deviation
        // The standard deviation is a measure of the amount of variation or dispersion of a set of values.
        double stdDeviation = Math.Sqrt(data.Average(v => Math.Pow(v - mean, 2)));
        // Calculate the minimum value
        // The minimum value is the smallest value in a data set.
        double min = data.Min();
        // Calculate the maximum value
        // The maximum value is the largest value in a data set.
        double max = data.Max();
        Console.WriteLine("Mean: " + mean);
        Console.WriteLine("Median: " + median);
        Console.WriteLine("Mode: " + mode);
        Console.WriteLine("Standard Deviation: " + stdDeviation);
        Console.WriteLine("Min: " + min);
        Console.WriteLine("Max: " + max);

    }
}