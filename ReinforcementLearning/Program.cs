using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System;

namespace TicTacToe
{
    public class TicTacToeEnv
    {
        private readonly char[,] board;
        private readonly char empty = '-';
        private readonly char playerX = 'X';
        private readonly char playerO = 'O';
        private char currentPlayer;

        // Setup an empty 3 x 3 board and set the current player to X
        public TicTacToeEnv()
        {
            board = new char[3, 3];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    board[i, j] = empty;
                }
            }
            currentPlayer = playerX;
        }

        public void DisplayBoard()
        {
            for (int i = 0; i < 3; i++)
            {

                for (int j = 0; j < 3; j++)
                {
                    Console.Write(board[i, j] + " ");
                }
                Console.WriteLine();
            }
        }

        // Make a move on the board if the cell is empty
        public bool MakeMove(int row, int col)
        {
            if (row >= 0 && row < 3 && col < 3 && board[row, col] == empty)
            {
                board[row, col] = currentPlayer;
                currentPlayer = currentPlayer == playerX ? playerO : playerX;
                return true;
            }
            return false;

        }

        // Check if the current player has won (horizontal, vertical, diagonal)
        public bool CheckWin(char player)
        {
            for (int i = 0; i < 3; i++)
            {
                if (board[i, 0] == player && board[i, 1] == player && board[i, 2] == player)
                {
                    return true;
                }
                if (board[0, i] == player && board[1, i] == player && board[2, i] == player)
                {
                    return true;
                }
            }
            if (board[0, 0] == player && board[1, 1] == player && board[2, 2] == player)
            {
                return true;
            }
            if (board[0, 2] == player && board[1, 1] == player && board[2, 0] == player)
            {
                return true;
            }
            return false;
        }

        // Check if the game is a draw and no more moves can be made
        public bool CheckDraw()
        {
            foreach (var cell in board)
            {
                if (cell == empty)
                {
                    return false;
                }
            }
            return true;
        }

        public char[,] GetBoard()
        {
            return board;
        }
    }

    public class QLearningAgent
    {
        private readonly Dictionary<string, double[]> qTable;
        private readonly double learningRate = 0.1;
        private readonly double discountFactor = 0.9;
        private readonly double explorationRate = 0.1;
        private readonly Random random = new Random();

        public QLearningAgent(double learningRate, double discountFactor, double explorationRate)
        {
            // A dictionary that maps states to arrays of Q-values. Each state is represented as a string, and the Q-values are stored as an array of doubles.
            qTable = new Dictionary<string, double[]>();
            this.learningRate = learningRate;
            this.discountFactor = discountFactor;
            this.explorationRate = explorationRate;
            random = new Random();
        }


        // Choose an action based on the current state. If the state is not in the Q-table, choose a random action.
        public int ChooseAction(char[,] board)
        {
            var state = GetState(board);
            if (!qTable.ContainsKey(state) || random.NextDouble() < explorationRate)
            {
                return random.Next(9);
            }

            var qValues = qTable[state];
            double maxQValue = double.MinValue;
            int action = 0;
            for (int i = 0; i < qValues.Length; i++)
            {
                if (qValues[i] > maxQValue)
                {
                    maxQValue = qValues[i];
                    action = i;
                }
            }

            return action;
        }


        // Update the Q-table based on the current state, action, reward, and the next state.
        public void UpdateQTable(char[,] board, int action, double reward, char[,] nextBoard)
        {
            var state = GetState(board);
            var nextState = GetState(nextBoard);
            if (!qTable.ContainsKey(state))
            {
                qTable[state] = new double[9];
            }
            if (!qTable.ContainsKey(nextState))
            {
                qTable[nextState] = new double[9];
            }

            double maxNextQValue = double.MinValue;
            foreach (var qValue in qTable[nextState])
            {
                if (qValue > maxNextQValue)
                {
                    maxNextQValue = qValue;
                }
            }

            qTable[state][action] = (1 - learningRate) * qTable[state][action] + learningRate * (reward + discountFactor * maxNextQValue);
        }

        // Get the state of the board as a string
        private string GetState(char[,] board)
        {
            char[] state = new char[9];
            int index = 0;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    state[index++] = board[i, j];
                }
            }

            return new string(state);
        }
    }

    public class Program
    {

        //  We are training a Tic-Tac-Toe playing agent using the Q-learning algorithm. The agent learns to make optimal moves by updating a Q-table based on the rewards it receives during gameplay.
        static void Main(string[] args)
        {
            var env = new TicTacToeEnv();
            var agent = new QLearningAgent(0.1, 0.9, 0.1);

            // This loop represents the training phase of the agent. The agent plays 10,000 games of Tic-Tac-Toe against a random opponent and updates its Q-table based on the rewards it receives.
            for (int i = 0; i < 10000; i++)
            
            {
                // Reset the environment for a new game
                env = new TicTacToeEnv();
                // The game loop, here '-' indicates that no winner has been determined yet
                char winner = '-';
                // Continue playing the game until a winner is determined
                while (winner == '-')
                {
                    // The agent chooses an action based on the current state of the board
                    var board = env.GetBoard();
                    //We use the ChooseAction method of the QLearningAgent class to select an action (move) based on the current state of the game board.
                    //If the current state is not in the Q-table or a random exploration is chosen, a random action is selected.
                    int action = agent.ChooseAction(board);
                    //We convert the selected action into row and column indices using simple arithmetic operations.
                    int row = action / 3;
                    int col = action % 3;
                    //We make the move on the board using the MakeMove method of the TicTacToeEnv class.
                    //If the move is valid, we proceed to check for a win or draw condition.

                    if (env.MakeMove(row, col))
                    {
                        var nextBoard = env.GetBoard();
                        if (env.CheckWin('X'))
                        {
                            winner = 'X';
                            //If the agent wins, we update the Q-table based on the current state, action, reward, and the next state using the UpdateQTable method of the QLearningAgent class.
                            //The reward for a win is typically set to 1.
                            agent.UpdateQTable(board, action, 1, nextBoard);
                        }
                        else if (env.CheckDraw())
                        {
                            winner = 'D';
                            //If the game is a draw, we update the Q-table based on the current state, action, reward, and the next state using the UpdateQTable method of the QLearningAgent class.
                            //The reward for a draw is typically set to 0.
                            agent.UpdateQTable(board, action, 0, nextBoard);
                        }
                        //If the game is not over, we update the Q-table based on the current state, action, reward, and the next state using the UpdateQTable method of the QLearningAgent class.
                        else
                        {
                            // The opponent makes a random move
                            while (true)
                            {
                                int opponentAction = new Random().Next(9);
                                int opponentRow = opponentAction / 3;
                                int opponentCol = opponentAction % 3;
                                if (env.MakeMove(opponentRow, opponentCol))
                                {
                                    break;
                                }
                            }

                            // Get the updated game board after opponent's move
                            nextBoard = env.GetBoard();
                            //If the opponent ('O') wins the game, we update the Q-table with a reward of -1 for the chosen action.
                            if (env.CheckWin('O'))
                            {
                                winner = 'O';
                                agent.UpdateQTable(board, action, -1, nextBoard);
                            }
                            //If the game is a draw, we update the Q-table with a reward of 0 for the chosen action.
                            else if (env.CheckDraw())
                            {
                                winner = 'D';
                                agent.UpdateQTable(board, action, 0, nextBoard);
                            }
                            //If the game is still ongoing, we update the Q-table with a reward of 0 for the chosen action.
                            else
                            {
                                agent.UpdateQTable(board, action, 0, nextBoard);
                            }
                        }
                    }


                    }
            }

            // After training the agent, we can test its performance by playing a game against it.
            var testEnv = new TicTacToeEnv();

            // The game loop for testing the agent against a random opponent
            while (true)
            {
                // Display the current state of the game board
                testEnv.DisplayBoard();
                var testBoard = testEnv.GetBoard();
                // The agent chooses an action based on the current state of the board
                int testAction = agent.ChooseAction(testBoard);
                int testRow = testAction / 3;
                int testCol = testAction % 3;
                testEnv.MakeMove(testRow, testCol);
                // Check if the agent ('X') has won or if the game is a draw and we break out of the loop
                if (testEnv.CheckWin('X') || testEnv.CheckDraw())
                {
                    testEnv.DisplayBoard();
                    break;
                }

                //Simulate the opponent's move by choosing a random valid action and making the move on the game board.
                while (true)
                {
                    int opponentAction = new Random().Next(9);
                    int opponentRow = opponentAction / 3;
                    int opponentCol = opponentAction % 3;
                    if (testEnv.MakeMove(opponentRow, opponentCol))
                    {
                        break;
                    }
                }

                //If the opponent ('O') wins the game or the game ends in a draw, we display the final state of the game board and break out of the testing loop.
                if (testEnv.CheckWin('O') || testEnv.CheckDraw())
                {
                    testEnv.DisplayBoard();
                    break;

                }
            }
            }
    }

}