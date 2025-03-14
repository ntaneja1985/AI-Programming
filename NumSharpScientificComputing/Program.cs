using MathNet.Numerics.Statistics;
using NumSharp;
using NumSharp.Utilities;
using System;
using System.Net.WebSockets;

public class Program
{
    public static void Main()
    {
        //NumSharpIntro();
        //NumSharpArrays();
        //MathOperations();
        //IndexingSlicingTechniques();
        //LinearAlgebra();
        //StatisticalAnalysis();
        ArrayBroadCastingUniversalFunctions();
    }

    public static void NumSharpIntro()
    {
        var array1D = np.array(new int[] { 1, 2, 3, 4, 5 });
        Console.WriteLine("1D Array");
        Console.WriteLine(array1D.ToString());

        var array2D = np.array(new int[,] { { 1, 2 }, { 3, 4 },{ 5, 6 } });

        Console.WriteLine("2D Array");
        Console.WriteLine(array2D.ToString());

        var array = np.array(new int[] { 1, 2, 3, 4, 5,6 });
        var reshapredArray = array.reshape(2, 3);
        Console.WriteLine("Reshaped Array");
        Console.WriteLine(array.ToString());

        var slicedArray = array["1:5"];
        Console.WriteLine("Sliced Array");
        Console.WriteLine(slicedArray.ToString());

        //Do addition
        var array1 = np.array(new int[] { 1, 2, 3 });
        var array2 = np.array(new int[] { 4, 5, 6 });
        var sumArray = array1 + array2;
        Console.WriteLine("Summed Array");
        Console.WriteLine(sumArray.ToString());

        //Do multiplication
        var productArray = array1 * array2;
        Console.WriteLine("Product Array");
        Console.WriteLine(productArray.ToString());

        //Linear Algebra
        var matrix1 = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var matrix2 = np.array(new double[,] { { 5,6 }, { 7,8 } });

        var matrixProduct = np.dot(matrix1, matrix2);
        Console.WriteLine("Matrix Product");
        Console.WriteLine(matrixProduct.ToString());
    }

    public static void NumSharpArrays()
    {
        Console.WriteLine("NumSharp array of 2s");
        NDArray numSharpArray = np.full(2, 5);
        foreach(var item in numSharpArray)
        {
            Console.WriteLine(item.ToString());
        }

        Console.WriteLine("NumSharp array of 0s");
        NDArray numSharpArrayZero = np.zeros(5);
        foreach (var item in numSharpArrayZero)
        {
            Console.WriteLine(item.ToString());
        }

        var array1D = np.array(new int[] { 1, 2, 3,4,5 });
        array1D[0] = 9;
        Console.WriteLine(array1D.ToString());

        var array2D = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        array2D[0][0] = 7;
        array2D[2][1] = 0;
        Console.WriteLine("2D Array");
        Console.WriteLine(array2D.ToString());

        var slicedArray = array1D["1:4"];
        Console.WriteLine("Sliced Array");
        Console.WriteLine(slicedArray.ToString());

        var array = np.array(new int[] { 1, 2, 3, 4, 5, 6 });
        var reshapredArray = array.reshape(2, 3);
        Console.WriteLine("Reshaped Array 2 x 3");
        Console.WriteLine(reshapredArray.ToString());

        var reshapredArray2 = array.reshape(3, 2);
        Console.WriteLine("Reshaped Array 3 x2");
        Console.WriteLine(reshapredArray2.ToString());
    }

    public static void MathOperations()
    {
        //Do addition
        var array1 = np.array(new int[] { 1, 2, 3,4,5 });
        var array2 = np.array(new int[] { 5, 4, 3,2,1 });
        var sumArray = array1 + array2;
        Console.WriteLine("Summed Array");
        Console.WriteLine(sumArray.ToString());

        var difference = array1 - array2;
        Console.WriteLine("Subtraction Array");
        Console.WriteLine(difference.ToString());

        var productArray = array1 * array2;
        Console.WriteLine("Product Array");
        Console.WriteLine(productArray.ToString());

        var quotient = array1 / array2;
        Console.WriteLine("Division of Array");
        Console.WriteLine(quotient.ToString());

        var array = np.array(new double[] { 1.0, 4.0, 9.0, 16.0 });
        var sqrtArray = np.sqrt(array);
        Console.WriteLine("Square Root of Array");
        Console.WriteLine(sqrtArray.ToString());

        var expArray = np.exp(array);
        Console.WriteLine("Exponential of Array");
        Console.WriteLine(expArray.ToString());

        var logArray = np.log(array);
        Console.WriteLine("Logarithm of Array");
        Console.WriteLine(logArray.ToString());


        var maxArray = np.maximum(array1, array2);
        Console.WriteLine("Elementwise Maximum");
        Console.WriteLine(maxArray.ToString());

        var minArray = np.minimum(array1, array2);
        Console.WriteLine("Elementwise Minimum");
        Console.WriteLine(minArray.ToString());
        
        //1,4,9,16,25
        var powerArray = np.power(array1, 2);
        Console.WriteLine("Elementwise Power(array1^2)");
        Console.WriteLine(powerArray.ToString());

    }

    public static void IndexingSlicingTechniques()
    {
        //Indexing arrays
        var array = np.array(new int[] { 10, 20, 30, 40, 50 });
        Console.WriteLine("First Element");
        Console.WriteLine(array[0].ToString());
        Console.WriteLine("Last Element");
        Console.WriteLine(array[-1].ToString());

        //Slicing Arrays
        var slice1 = array["1:4"];
        Console.WriteLine("Slice from index 1 to 3");
        Console.WriteLine(slice1.ToString());

        var slice2 = array[":3"];
        Console.WriteLine("Slice from start to index 2:");
        Console.WriteLine(slice2.ToString());

        var slice3 = array["2:"];
        Console.WriteLine("Slice from index 2 to end:");
        Console.WriteLine(slice3.ToString());


        var array2D = np.array(new int[,] { { 1, 2,3 }, { 4,5,6 }, { 7,8,9 } });
        Console.WriteLine("Element at (1,1) 2D Array");
        Console.WriteLine(array2D[1,1].ToString());

        var slice2D = array2D["1:,:2"];
        Console.WriteLine("Slice from row 1 to end and columns 0 to 1");
        Console.WriteLine(slice2D.ToString());

    }

    public static void LinearAlgebra()
    {
        var matrixA = np.array(new double[,] { { 1, 2 },{ 3, 4 } });
        var matrixB = np.array(new double[,] { {5,6 },{ 7, 8 } });

        Console.WriteLine("MatrixA * MatrixB");
        //Cartesian Product
        var result = np.dot(matrixA,matrixB);
        Console.WriteLine(result.ToString());
    }

    public static void StatisticalAnalysis()
    {
        var data = np.array(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

        //Get the descriptive statistics
        var mean = np.mean(data);
        var stdDev = np.std(data);
        var min = np.min(data);
        var max = np.max(data);

        //Get the covariance
        //Covariance is a statistical measure that indicates the degree to which two random variables change together.
        //If the covariance is positive, it suggests that as one variable increases, the other tends to increase as well.
        //If it is negative, one variable tends to decrease as the other increases.

        int n = 5;
        var sample1 = new double[] {1.1,2.2,3.3,4.4,5.5 };
        var sample2 = new double[] { 2.1, 3.2, 4.3, 5.4, 6.5 };

        var covariance = Statistics.Covariance(sample1 , sample2);
        Console.WriteLine($"Covariance is: {covariance}");

    }

    public static void ArrayBroadCastingUniversalFunctions()
    {
        var array1 = np.array(new double[] { 1, 2, 3 });
        var array2 = np.array(new double[,] { { 1}, { 2 },{ 3 } });

        var result = array1 + array2;
        Console.WriteLine("Broadcasted addition result ");
        //Broadcasted addition result
        //[[2, 3, 4],
        //[3, 4, 5],
        //[4, 5, 6]]
        Console.WriteLine(result.ToString());


        // Elementwise operations on arrays using ufuncs
        // Ufuncs provide a concise and efficient way to apply operations to entire arrays.
        // Combining broadcasting and ufuncs can greatly simplify complex array operations.
        var squared = np.power(array1, 2);
        var sqrt = np.sqrt(array1);
        Console.WriteLine(squared.ToString());
        Console.WriteLine(sqrt.ToString());


        var array3 = np.array(new double[,] { { 1,2,3 }, { 4,5,6 } });
        var array4 = np.array(new double[] { 1, 2, 3 } );

        var result2 = np.power(array3 + array4 , 2);
        //Broadcasting and UFunc combined
        Console.WriteLine(result2.ToString());

        var array5 = np.arange(12);
        var array6 = np.reshape(array5,new Shape(3,4));
        Console.WriteLine("Original Array");
        //Original Array
        //[0, 1, 2, 3, 4, ..., 7, 8, 9, 10, 11]
        Console.WriteLine(array5.ToString());
        //Reshaped array
        //[[0, 1, 2, 3],
        //[4, 5, 6, 7],
        //[8, 9, 10, 11]]
        Console.WriteLine("Reshaped array");
        Console.WriteLine(array6.ToString());


        //Stacking Arrays(combining arrays)
        //Stacking arrays combines multiple arrays into a single array along a specified axis.
        //We can stack arrays horizontally using NP dot stack, or vertically using NP dot stack.

        var array7 = np.array(new int[,] { {1,2},{ 3, 4 } });
        var array8 = np.array(new int[,] { { 5, 6 }, { 7,8 } });

        var stackedHorizontally = np.hstack([array7, array8]);
        var stackedVertically = np.vstack([array7, array8]);
        Console.WriteLine("Stacked Horizontally:");
        Console.WriteLine(stackedHorizontally.ToString());
        Console.WriteLine("Stacked Vertically:");
        Console.WriteLine(stackedVertically.ToString());


    }
}