using MathNet.Numerics;
using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Differentiation;
using MathNet.Numerics.Integration;
using MathNet.Numerics.LinearAlgebra.Factorization;
using MathNet.Numerics.Interpolation;
using MathNet.Numerics.Optimization;
using MathNet.Numerics.LinearAlgebra.Double.Solvers;


class Program
{
    static void Main()
    {
        //ManipulateVectorsAndMatrices();
        //WorkWithStatisticAnalysisInMathNetNumerics();
        //WorkWithLinearAlgebraOperations();
        //NumericalIntegrationAndDifferentiation();
        //LinearEquationsAndSystems();
        //CurveFittingInterpolationTechniques();
        //Eigen Value decomposition
        //EigenValueDecomposition();
        MultiVariateDataAnalysisAndDimensionalityReduction();
    }

    // Manipulate Vectors and Matrices
    static void ManipulateVectorsAndMatrices()
    {
        var vector = Vector<double>.Build.DenseOfArray(new double[] { 1.0, 2.0, 3.0 });
        Console.WriteLine("Vector: " + vector);
        Console.WriteLine("First Element: " + vector[0]);
        var scaledVector = vector * 2.0;
        Console.WriteLine("Scaled Vector: " + scaledVector);
        var matrix = Matrix<double>.Build.DenseOfArray(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } });
        Console.WriteLine("Matrix: " + matrix);
        Console.WriteLine("Element at 0,0: " + matrix[0, 0]);
        Console.WriteLine("Element at 0,1: " + matrix[0, 1]);
        Console.WriteLine("Element at 1,2: " + matrix[1, 2]);

        var addedMatrix = matrix + 1.0;
        Console.WriteLine("Added Matrix: " + addedMatrix);
    }

    static void WorkWithStatisticAnalysisInMathNetNumerics()
    {
        double[] data = new double[] { 1.2, 2.3, 3.4, 4.5, 5.6,6.7 };
        var mean = Statistics.Mean(data);
        var stdDev = Statistics.StandardDeviation(data);
        var median = Statistics.Median(data);
        var min = Statistics.Minimum(data);
        var max = Statistics.Maximum(data);

        Console.WriteLine("Mean: " + mean);
        Console.WriteLine("Standard Deviation: " + stdDev);
        Console.WriteLine("Median: " + median);
        Console.WriteLine("Min: " + min);
        Console.WriteLine("Max: " + max);


        
        var normalDist = new MathNet.Numerics.Distributions.Normal(mean, stdDev);

        var prob = normalDist.CumulativeDistribution(3.0);
        Console.WriteLine("Probability of value being less than 3.0: " + prob);

        var tTest = new StudentT();
        var pValue = tTest.CumulativeDistribution(prob);
        Console.WriteLine("P-Value for the paired t-Test: " + pValue);

    }

    static void WorkWithLinearAlgebraOperations()
    {
        var matrixA = DenseMatrix.OfArray(new double[,]
        {
            {1,2 },
            {3,4}

        });
        var matrixB = DenseMatrix.OfArray(new double[,]
       {
            {5,6 },
            {7,8}

       });

        var matrixAdd = matrixA + matrixB;
        Console.WriteLine("Matrix Addition:\n" + matrixAdd);

        var matrixSub = matrixA - matrixB;
        Console.WriteLine("Matrix Subtraction: " + matrixSub);

        //Cartesian Product of the matrix
        var matrixMul = matrixA * matrixB;
        Console.WriteLine("Matrix Multiplication: " + matrixMul);

        /*
          The inverse of a matrix is a new matrix that, when multiplied with the original matrix, results in the identity matrix. 
          In other words, if A is the original matrix and A^-1 is its inverse, then A * A^-1 = I, where I is the identity matrix.
          Inverse of matrix A is   -2   1
                                    1  -0.5
          To verify that the inverse is correct, we can multiply matrixA with matrixInv:
          Result is the identity matrix: 
          1  0
          0  1
         */
        var matrixInv = matrixA.Inverse();
        Console.WriteLine("Matrix Inversion\n" + matrixInv);

        var b = DenseVector.OfArray(new double[] { 1, 2 });
        /*
         The Solve method is called on the matrixA object. The Solve method solves the linear system of equations Ax = b, where A is the matrix and x is the unknown vector. The result of the Solve method is assigned to the variable x.
         This means we have to solve this equation: 
        1x + 2y = 1
        3x + 4y = 2
        Here x = 0 and y = (1 - 1x)/2
        So y = 0.5
         */
        var x = matrixA.Solve(b);
        Console.WriteLine("Solution to Ax = b:\n"+x);
    }

    static void NumericalIntegrationAndDifferentiation()
    {
        Func<double,double> function = x => Math.Sin(x);
        /*
          This line of code is performing numerical integration using the Simpson's rule. It calculates the integral of a given function over a specified range.
          In the provided code, the SimpsonRule.IntegrateComposite method is called with the following parameters:
            •	function: This is a lambda function that represents the function to be integrated. In this case, it is x => Math.Sin(x), which represents the sine function.
            •	0: This is the lower bound of the integration range.
            •	Math.PI: This is the upper bound of the integration range.
            •	1000: This is the number of intervals used in the composite Simpson's rule.
          The result of the integration is stored in the integral variable.
          It allows us to calculate the definite integral of a function over a specified interval using Simpson's rule. 
          This is important in various fields such as physics, engineering, and economics where integration is used to determine quantities like area under a curve, total accumulated value, and more.
        1.	Pharmacokinetics: The area under the concentration-time curve (AUC) of a drug in the bloodstream helps in understanding the drug's absorption, distribution, metabolism, and excretion. This is crucial for determining appropriate dosages.
        2.	Medical Imaging: In techniques like MRI and CT scans, integrating signal intensities can help in reconstructing images of the body's interior, aiding in diagnosis and treatment planning.
         */

        double integral = SimpsonRule.IntegrateComposite(function, 0, Math.PI, 1000);
        Console.WriteLine($"Numerical Integration of sin(x) from 0 to pi:  "+integral);

        /*
         Numerical derivatives are useful because they allow us to approximate the derivative of a function when an analytical solution is difficult or impossible to obtain. 
        This is particularly helpful in real-world applications where functions may not have simple closed-form expressions or where data is noisy or discrete.
         •	Rate of Change: Numerical derivatives can be used to calculate the rate of change of economic indicators, such as inflation rates or stock prices.
         •	Gradient Descent: Numerical derivatives are used to compute gradients in optimization algorithms, which are essential for training machine learning models.
         •	Sensitivity Analysis: They help in understanding how changes in input variables affect the output of a model.
         •	Signal Processing: If sin(x) represents a signal, its derivative can provide information about the rate of change of the signal, which is useful in filtering and analyzing the signal.
         •	Control Systems: In control theory, understanding the rate of change of a system's output can help in designing controllers that respond appropriately to changes in the system.
         */
        Func<double,double> functionToDifferentiate = x => Math.Sin(x);
        NumericalDerivative derivative = new NumericalDerivative();
        double derivativeAtPoint = derivative.EvaluateDerivative(functionToDifferentiate, 1, 1);
        Console.WriteLine($"Numerical Derivative of sin(x) at x = 1: {derivativeAtPoint}");
    }

    static void LinearEquationsAndSystems()
    {
        /*
         This code is demonstrating the use of LU decomposition to solve a system of linear equations. 
         LU decomposition is a method that decomposes a square matrix into the product of a lower triangular matrix and an upper triangular matrix.
         It is commonly used to solve systems of linear equations efficiently.
         3.	The code calls the LU method on the matrix A. This method performs LU decomposition on the matrix A and returns an LU factorization object.
        4.	The code calls the Solve method on the LU factorization object, passing in the vector b. 
            This method solves the system of linear equations Ax = b, where A is the matrix and x is the unknown vector. 
            The result of the Solve method is assigned to the variable x.
            LU decomposition is a powerful technique for solving systems of linear equations and is widely used in various fields of mathematics, science, and engineering.
         */
        var A = DenseMatrix.OfArray(new double[,] { {3,2,-1 },{2,-2,4 },{-1,0.5,-1 } });
        var b = DenseVector.OfArray(new double[] {1,-2,0} );

        var lu = A.LU();
        var x = lu.Solve(b);

        Console.WriteLine($"Solution using LU Decomposition: "+ x);

        /*
         This code demonstrates the use of QR decomposition to solve an overdetermined system of linear equations. 
        QR decomposition is a method that decomposes a matrix into the product of an orthogonal matrix and an upper triangular matrix. 
        It is commonly used to solve systems of linear equations efficiently.
         */
        var A_over = DenseMatrix.OfArray(new double[,]
        {
            {1, 1 },
            {2, 3 },
            {4,5 } });

        var B_over = DenseVector.OfArray(new double[] {6,14,24} );

        var qr = A_over.QR();
        var x_over = qr.Solve(B_over);
        Console.WriteLine($"Solution to overdetermined system using QR decomposition:" + x_over);


        /*
         The selected code demonstrates how to solve an underdetermined system of linear equations using Singular Value Decomposition (SVD). 
        This is useful in various scenarios where you have more unknowns than equations, making the system underdetermined. Here are some practical applications:
        1.	Data Science and Machine Learning: In these fields, you often encounter situations where you have more features (variables) than samples (equations). SVD can help in dimensionality reduction and solving such systems.
        2.	Signal Processing: SVD is used in signal processing for noise reduction and data compression.
        3.	Control Systems: In control theory, SVD can be used to design controllers for systems with more control inputs than outputs.
        4.	Image Processing: SVD is used in image compression techniques like JPEG.
        5.	Economics and Finance: In these fields, SVD can be used for portfolio optimization and risk management when dealing with large datasets.
        This approach ensures that you can find a solution even when the system does not have a unique solution, which is common in real-world applications.

         */
        var A_under = DenseMatrix.OfArray(new double[,]
        {
            {2,3,1 },
            {1,1,0 },
             });

        var B_under = DenseVector.OfArray(new double[] { 1,2 });

        var svd = A_under.Svd(true);
        var x_under = svd.Solve(B_under);
        Console.WriteLine($"Solution to underdetermined system using SVD:" + x_under);

    }

    static void CurveFittingInterpolationTechniques()
    {
        /*
         This code is performing polynomial curve fitting using the MathNet.Numerics library. 
         Polynomial curve fitting is a technique used to find a polynomial function that best fits a given set of data points.
         In this code, the xData array represents the x-coordinates of the data points, and the yData array represents the corresponding y-coordinates. The Fit.Polynomial method is used to fit a polynomial function to the data points.
         This code is useful when you have a set of data points and want to find a polynomial function that closely approximates the relationship between the x and y values. Polynomial curve fitting is commonly used in various fields, such as data analysis, signal processing, and machine learning.
         Polynomial curve fitting has various applications in different fields. Here are some common applications:
         Data Analysis: Polynomial curve fitting is often used in data analysis to model and approximate relationships between variables. It can help identify trends, patterns, and make predictions based on the given data.
         */
        double[] xData = { 1, 2, 3, 4, 5, };
        double[] yData = { 1, 4, 9, 16, 25 };
        var polyFit = Fit.Polynomial(xData, yData, 2);
        Console.WriteLine("Polynomial Coefficient");
        foreach(var coeff in polyFit)
        {
            Console.WriteLine(coeff.ToString());
        }

        //This code is evaluating a polynomial function at a specific value of x and printing the result.
        double polyValue = Polynomial.Evaluate(6);
        Console.WriteLine($"Polynomial Value at x= 6: {polyValue}");


        /*
          This code is performing linear interpolation using the MathNet.Numerics library. 
        Linear interpolation is a method used to estimate values between two known data points. 
        In this code, the Interpolate.Linear method is used to create a linear interpolation function based on the provided xData and yData arrays. 
        The Interpolate object returned by Interpolate.Linear represents the linear interpolation function.
         */
        var linearInterp = Interpolate.Linear(xData, yData);
        double linearValue = linearInterp.Interpolate(2.5);
        Console.WriteLine($"Linear interpolation at x = 2.5: {linearValue}");


        /*
        This code is performing cubic spline interpolation using the MathNet.Numerics library. 
        Interpolation is a method used to estimate values between known data points. 
        In this code, the Interpolate.CubicSpline method is used to create a cubic spline interpolation function based on the provided xData and yData arrays.
        Cubic spline interpolation is a technique that uses piecewise-defined cubic polynomials to approximate a smooth curve that passes through the given data points. 
        It provides a more accurate and smooth interpolation compared to linear interpolation.
         */
        var splineInterp = Interpolate.CubicSpline(xData, yData);
        double splineValue = splineInterp.Interpolate(2.5);
        Console.WriteLine($"Cubic spline interpolation at x = 2.5: {splineValue}");

    }

    static void OptimizationMethods()
    {
        /*
         This code is performing optimization using the BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm in the MathNet.Numerics library. 
         Optimization is the process of finding the best solution (minimum or maximum) for a given objective function, often subject to certain constraints.
         Optimization algorithms are used to minimize the loss function during the training of machine learning models, such as neural networks.
         In machine learning, the BFGS algorithm can be used to optimize the weights of a neural network. 
         The objective function in this case would be the loss function, which measures the difference between the predicted and actual values. 
         The gradient function would compute the gradient of the loss function with respect to the weights.
         By using optimization techniques like BFGS, you can efficiently find the best parameters for your models and systems, leading to improved performance and outcomes in various applications.
 
         */
        Func<Vector<double>,double> objectiveFunction = x => Math.Pow(x[0], 2) + Math.Pow(x[1],2);
        Func<Vector<double>, Vector<double>> gradientFunction = x => Vector<double>.Build.DenseOfArray(new double[] { 2 * x[0], 2 * x[1] });
        var solver = new BfgsMinimizer(1e-6, 100, 1);
        var result = solver.FindMinimum(ObjectiveFunction.Gradient(objectiveFunction, gradientFunction), Vector<double>.Build.DenseOfArray([1.0, 1.0]);
        Console.WriteLine($"Optimal Point: {result.MinimizingPoint}");
        Console.WriteLine($"Optimal Value: {result.FunctionInfoAtMinimum.Value}");
    }

    static void SparseMatrixRepresentation()
    {
        int rows = 4;
        int columns = 4;
        var sparseMatrix = SparseMatrix.OfIndexed(rows, columns, new[]
        {
            Tuple.Create(0,0,1.0),
            Tuple.Create(1,1,2.0),
            Tuple.Create(2,2,3.0),
            Tuple.Create(3,3,4.0),
            Tuple.Create(0,3,5.0)
        });

        Console.WriteLine("Sparse Matrix (CSR Format):");
        Console.WriteLine(sparseMatrix);

        var denseMatrix = DenseMatrix.OfIndexed(rows, columns, new[]
        {
            Tuple.Create(0,0,1.0),
            Tuple.Create(1,1,2.0),
            Tuple.Create(2,2,3.0),
            Tuple.Create(3,3,4.0),
            Tuple.Create(0,3,5.0)
        });

        var result = sparseMatrix.Multiply(denseMatrix);
        Console.WriteLine("Result of multiplication with Dense Matrix");
        Console.WriteLine(result);
    }

    static void EigenValueDecomposition()
    {
        var matrix = DenseMatrix.OfArray(new double[,]
        {
            {4, 2 },
            {1,1 }
        });

        var evd = matrix.Evd();

        var eigenValues = evd.EigenValues;
        var eigenVectors = evd.EigenVectors;

        Console.WriteLine($"Eigen Values: {eigenValues}");
        Console.WriteLine($"Eigen Vectos: {eigenVectors}");


        var matrixSvd = DenseMatrix.OfArray(new double[,]
        {
            {1,0,0,0,2 },
            {0,0,3,0,0 },
            {0,0,0,0,0 },
            {0,4,0,0,0 }
        });

        var svd = matrixSvd.Svd();
        var U = svd.U;
        var S = svd.S;
        var VT = svd.VT;

        Console.WriteLine($"U Value: {U} S Value: {S} VT Value: {VT}");
    }

    static void MultiVariateDataAnalysisAndDimensionalityReduction()
    {
        var dataArray = new double[,]
        {
            {2.5,2.4 },
             {0.5,0.7 },
              {2.2,2.9 },
               {1.9,2.2 },
                {3.1,3.0 },
                 {2.3,2.7 },
                  {2.0,1.6 },
                   {1.0,1.1 },
                    {1.5,1.6 },
                     {1.1,1.9 },
        };

        var dataMatrix = Matrix<double>.Build.DenseOfArray(dataArray);

        var columnMeans = dataMatrix.ColumnSums() / dataMatrix.RowCount;
        var centeredMatrix = dataMatrix - Matrix<double>.Build.Dense(dataMatrix.RowCount,dataMatrix.ColumnCount,(i,j) => columnMeans[j]);
        var covarianceMatrix = centeredMatrix.TransposeThisAndMultiply(centeredMatrix) / (dataMatrix.RowCount - 1);
        var evd = covarianceMatrix.Evd();
        var principalComponent = evd.EigenVectors;
        Console.WriteLine("Principal Components:");
        Console.WriteLine(principalComponent);
    }


}