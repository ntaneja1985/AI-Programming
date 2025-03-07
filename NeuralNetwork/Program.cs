
//This is a simple implementation of a neural network in C# that can perform the OR operation.
NeuralNetwork neuralNetwork = new NeuralNetwork();

//Specify the inputs for training the neural network
double[,] trainingInputs = new double[,]
{
    {0, 0, 0},
    {1, 1, 1},
    {1, 0, 0}
};

//Remember this will test the OR operation in the neural network
//The OR operation is a logical operation that takes two binary inputs and returns true (1) if at least one of the inputs is true (1), and false (0) otherwise.
double[,] trainingOutputs = new double[,]
{
    {0},
    {1},
    {1}
};

//Train the neural network with the training data
neuralNetwork.Train(trainingInputs, trainingOutputs, 1000);

//Test the neural network with new data
double[,] output = neuralNetwork.Think(new double[,] {
    { 0, 1, 0 },
    { 0, 0, 0 },
    { 0, 0, 1 }

});

//Print the output of the neural network
PrintMatrix(output);

//Method to print a 2D array
static void PrintMatrix(double[,] matrix)
{
    int rows = matrix.GetLength(0);
    int cols = matrix.GetLength(1);
    for(int row = 0; row < rows; row++)
    {
        for(int column = 0; column < cols; column++)
        {
            Console.Write(Math.Round(matrix[row,column]) + " ");
        }

        Console.WriteLine();
    }
}

//This class represents a simple implementation of a neural network in C# that can perform the OR operation.
//The neural network is trained using a set of input-output pairs and then used to make predictions on new data.
public class NeuralNetwork
{
    //2D array to store the weights of the neural network
    //In a neural network, the weights represent the strength of the connections between the nodes.
    //Each weight corresponds to a connection between two nodes.
    //A 2D array is used to store the weights in a neural network because it allows for a flexible and efficient representation of the connections.
    //The first dimension of the array represents the input nodes, and the second dimension represents the output nodes.
    //Each element in the array represents the weight of the connection between a specific input node and a specific output node.
    // By using a 2D array, we can easily access and manipulate the weights for each connection in the neural network.
    // For example, if we want to update the weight between the first input node and the second output node, we can simply access weights[0, 1] and modify its value.
    // Overall, using a 2D array for weights provides a structured and organized way to represent the connections in a neural network, making it easier to perform computations and update the weights during the training process.

    private double[,] weights;

    //Enum to represent the operations that the neural network can perform
    enum OPERATION { ADD, SUBTRACT, MULTIPLY };

    //Constructor to initialize the weights of the neural network
    public NeuralNetwork()
    {
        Random randomNumber = new Random();
        //Number of input nodes and output nodes
        int numberOfInputNodes = 3;
        int numberOfOutputNodes = 1;
        weights = new double[numberOfInputNodes, numberOfOutputNodes];
        //Initialize the weights with random values between -1 and 1
        for (int i = 0; i < numberOfInputNodes; i++)
        {
            for (int j = 0; j < numberOfOutputNodes; j++)
            {
                weights[i, j] = 2* randomNumber.NextDouble() - 1;
            }
        }
    }

    //Method to transpose a 2D array
    private double[,] Transpose(double[,] matrix)
    {
        return matrix.Cast<double>().ToArray().Transpose(matrix.GetLength(0), matrix.GetLength(1));
    }


    //Method to perform a feedforward operation in the neural network
    // the Activate method applies the sigmoid activation function to each element in the input matrix and returns the resulting matrix.
    // It also has the option to calculate the derivative of the sigmoid function if specified.
    // This method is an essential step in the feedforward process of a neural network, where the input values are transformed through activation functions to produce the network's output.
    private double[,] Activate(double[,] matrix, bool isDerivative)
    {
        int numberOfRows = matrix.GetLength(0);
        int numberOfCols = matrix.GetLength(1);
        double[,] result = new double[numberOfRows, numberOfCols];
        for (int row = 0; row < numberOfRows; row++)
        {
            for (int col = 0; col < numberOfCols; col++)
            {
                double sigmoidOutput = result[row,col] = 1/(1+ Math.Exp(-matrix[row,col]));
                double derivativeSigmoidOutput = result[row,col] = matrix[row,col] * (1 - matrix[row,col]);
                result[row,col] = isDerivative ? derivativeSigmoidOutput : sigmoidOutput;
            }
        }

        return result;
    }

    
    public void Train(double[,] trainingInputs, double[,] trainingOutputs, int numberOfIterations)
    {
        for (int iteration = 0; iteration < numberOfIterations; iteration++)
        {
            double[,] output = Think(trainingInputs);
            double[,] error = PerformOperation(trainingOutputs, output, OPERATION.SUBTRACT);
            double[,] adjustment = DotProduct(Transpose(trainingInputs), PerformOperation(error,Activate(output, true), OPERATION.MULTIPLY));
            weights = PerformOperation(weights, adjustment, OPERATION.ADD);
        }
    }


    // 
    private double[,] DotProduct(double[,] matrix1, double[,] matrix2)
    {
        int numberOfRowsInMatrix1 = matrix1.GetLength(0);
        int numberOfColsInMatrix1 = matrix1.GetLength(1);

        int numberOfRowsInMatrix2 = matrix2.GetLength(0);
        int numberOfColsInMatrix2 = matrix2.GetLength(1);

        double[,] result = new double[numberOfRowsInMatrix1, numberOfColsInMatrix2];
        for(int rowInMatrix1 = 0; rowInMatrix1 < numberOfRowsInMatrix1; rowInMatrix1++)
        {
            for (int colInMatrix2 = 0; colInMatrix2 < numberOfColsInMatrix2; colInMatrix2++)
            {
                double sum = 0;
                for (int colInMatrix1 = 0; colInMatrix1 < numberOfColsInMatrix1; colInMatrix1++)
                {
                    sum += matrix1[rowInMatrix1, colInMatrix1] * matrix2[colInMatrix1, colInMatrix2];
                }
                result[rowInMatrix1, colInMatrix2] = sum;
            }
        }

        return result;

    }

    //The PerformOperation method takes two matrices and an operation as input and performs the specified operation on each element of the matrices.
    //n a neural network, element-wise operations are commonly used during the training process to update the weights based on the calculated error. The PerformOperation method allows for flexible and efficient computation of element-wise operations, such as addition, subtraction, and multiplication.
    //By using a nested loop, the method iterates over each element of the matrices and performs the specified operation based on the OPERATION parameter. The result is stored in a new matrix, which is then returned.
    //For example, during the training process, the PerformOperation method is used to subtract the predicted output from the desired output to calculate the error. It is also used to multiply the error with the derivative of the sigmoid function to adjust the weights. These element-wise operations are essential for updating the weights and improving the performance of the neural network.
    private double[,] PerformOperation(double[,] matrix1, double[,] matrix2, OPERATION operation)
    {
        int numberOfRows = matrix1.GetLength(0);
        int numberOfCols = matrix1.GetLength(1);
        double[,] result = new double[numberOfRows, numberOfCols];
        for (int row = 0; row < numberOfRows; row++)
        {
            for (int col = 0; col < numberOfCols; col++)
            {
                switch (operation)
                {
                    case OPERATION.ADD:
                        result[row, col] = matrix1[row, col] + matrix2[row, col];
                        break;
                    case OPERATION.SUBTRACT:
                        result[row, col] = matrix1[row, col] - matrix2[row, col];
                        break;
                    case OPERATION.MULTIPLY:
                        result[row, col] = matrix1[row, col] * matrix2[row, col];
                        break;
                }
            }
        }
        return result;
    }

    public double[,] Think(double[,] inputs)
    {
        return Activate(DotProduct(inputs, weights), false);
    }

}

public static class Extensions
{
    //Extension method to transpose a 2D array
    //The Transpose method is an extension method that transposes a 2D array.
    //Transposing a matrix means converting its rows into columns and its columns into rows.
    //This is useful in various mathematical and computational operations.
   //The Transpose method achieves this by creating a new 2D array with the dimensions of the transposed matrix.
   //It then iterates over the original matrix and assigns the values to the corresponding positions in the transposed matrix.
    public static double[,] Transpose(this double[] array, int rows, int columns)
    {
        double[,] result = new double[columns, rows];
        for (int row = 0; row < rows;row++)
        {
            for (int col = 0; col < columns; col++)
            {
                result[col, row] = array[row * columns + col];
            }
        }
        return result;
    }
}
