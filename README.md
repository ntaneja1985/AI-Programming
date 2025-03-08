# AI Programming in C#
- using ML.NET 

- We programmers create an artificial brain 
- We reward good answers and punish bad answers 
- We want an AI to distinguish between apples and oranges 
- We train it on 500 pictures of apples and oranges 
- ![alt text](image.png)
- We then show pictures and ask it to guess if the image is an apple or orange 
- This example is a subtype of AI called machine learning and it is an image classification algorithm 
- ChatGPT is a generative AI 
- It is trained on super large datasets.  
- Here AI program generates a response based on an input. 
- In AI there are subfields 
- ![alt text](image-1.png)
- Other examples of AI are Alexa, Google Home and ChatGPT. 
- AI can also recognize emotions, however it cannot respond to them. 


## Main AI Concepts 

### Types of AI 
- Narrow AI and Strong AI 
- Narrow AI is used to perform specific tasks like an AI designed to play a game 
- Strong AI is a higher powered type of AI and can work on different types of tasks 

### Subtypes of AI 
- ![alt text](image-2.png)

- #### Machine Learning
- In Machine Learning, we use data to train an AI model. After the model is trained, we present it with a situation and it makes a prediction based on its training. 
- Machine learning is of 3 types: 
- Supervised Learning , Unsupervised Learning and Reinforcement Learning
- **Supervised learning** is when we use labelled data to train our model to recognize patterns and predict outcomes 
- **Unsupervised learning** uses data without any labels, which the model uses to discover patterns that exist in the data and then uses the inferences to predict the outcome 
- In **Reinforcement learning**, we assign rewards and punishments to tasks to train the model, when then is able to make predictions using those results. 
- #### Classification 
- Another subtype of AI where the model puts data into different categories based on certain features it has 
- For example, we can classify words or sentences based on if they have a negative or positive emotion in them
- We can also classify images like grouping images of different fruits 
- Types of Classification are :
- **Binary Classification**
- This means we only have 2 categories: like Apples and Oranges
- **Multiclass Classification**
- More than 2 categories like Apples, Oranges and Banana
- **MultiLabel classification** 
- When the same type of data belongs to more than one category. For example if our program has categories: "nature" and "large", then mountains can belong to both these categories 
- Lot of classification algorithms 
- #### Regression 
- It is a strategy that plots all of the known data points and based on how the data is trending, it can predict future values. It allows us to see the relationship between 2 variables. 
- One variable on the X-Axis and One on the Y-Axis. 
- #### Forecasting 
- When an AI can predict future events 
- #### Recommendation 
- An AI strategy which is used to make suggestions based on prior choices. 
- If we viewed items on online store like books on Amazon's website and then seeing recommendations of similar items, that is an example of this type of AI 
- #### Neural Networks 
- ![alt text](image-4.png)
- They are like digital brains. We write code that enables the program to learn from its mistakes based on the way that the human brain works 
- Just like brain has neurons, neural network has perceptrons, which are the digital version of neurons. 
- By adding multiple layers of interconnected perceptrons, we are able to build a neural network that mimics the functionality of a human brain by sending signals between perceptrons. 
- #### Q-Learning 
- Type of reinforcement learning where a model will perform a task over and over and improve taking the correct action over time. 
- #### Deep Learning 
- Enables AI to recognize complex learning and is made possible thanks to neural networks. Examples are deep Q-Learning and Deep Convolutional Q-Learning. 

## Neural Networks 
- A computer program that is able to learn and is modeled around how human brain works 
- Neural network are made up of digital equivalent of neurons called perceptrons. 
- ![alt text](image-5.png)
- Each perceptron can take input from one or more perceptrons and send output to one more more perceptrons 
- Neural networks are made up of layers in the most basic form of a neural network.
- There are 3 main layers:
- Input Layer: Made up of perceptrons that make simple decisions based on input 
- Hidden Layer: Perceptrons that make more complex decisions by weighing the results from the first layer 
- Output Layer: Produces the output of the program 
- ![alt text](image-6.png)
- It is possible for a neural network to have more than one hidden layer between the input layer and the output layer 
- ![alt text](image-7.png)
- When a neural network has multiple hidden layers, this enables it to make even more complex decisions. 
- It is called a deep neural network and it is able to work with massive amounts of data.
- Perceptrons are connected by connections and each connection has a weight associated with it. This determines how much influence each perceptron has on the other.  An activation function determines whether or not a perceptron should fire based on the sum of its inputs, weighted using the weight associated with the connection between the two.
- Learning occurs as the network gets input and gets better and better at producing the correct output, thanks to its adjusting its weights
- ![alt text](image-8.png)
- 3 types of neural networks:
- Feed Forward Neural Networks: Data flows in one direction from input to output node. Each node is connected to the next node in the layer.
- Backpropagation Neural Networks : It is a type of Feedforward neural network which gives feedback that is uses to improve its decisions
- Convolutional Neural Networks: It detects patterns in the data based on images and spatial data
- These types are named based on how data flows from the input node to the output node.
- Lets understand with an example 
- ![alt text](image-9.png)
- We can show a lot of handwritten numbers along with their correct values. Neural networks adjust their weights based on its understanding to get better at recognizing the numbers
- Once trained, a neural network can look at a new handwritten number and tell us what it thinks it is, just like human beings do. 

## Machine Learning 
- subfield of AI 
- Lets say we make an algorithm to add 2 numbers
- ![alt text](image-10.png)
- With machine learning, we do the opposite. We give the program the data and the expected results and then it comes up with the algorithm. 
- We can provide a pair of numbers and their solutions and it will infer from the data 
- In machine learning, we refer to the AI as the model and there are 2 phases a model goes through: 
- ![alt text](image-11.png)
- In training we provide data to train the model 
- 3 main types of ML
- ![alt text](image-13.png)
- In supervised learning the model builds a profile of an object based on its labels
- In unsupervised learning, it  builds a profile of similar looking objects by understanding the patterns between them and grouping them accordingly. 
- In reinforcement learning, we have a system of rewards and punishments. If the model correctly identifies the picture of an apple as an apple, we may reward it with a positive score. If it incorrectly labels, we provide a negative score. The model will try the max positive score as possible .

## Q-Learning
- Fundamental concept in reinforcement learning that focuses on learning the optimal action-selection strategy given a particular state of the environment. The agent learns to make decisions by interacting with its environment.
- Imagine playing a video where we control a character navigating through a maze. 
- ![alt text](image-14.png)
- At each step we can take multiple actions like moving up, down, left and right. 
- The goal is to reach the end of the maze while maximizing our score or reward. 
- In Q-Learning, we use a table called the Q-Table to keep track of the expected future rewards for each possible action in each state of the environment. 
- Each cell in the Q-table represents the "quality" or "value" of taking a particular action in a specific state.
- We start by initializing the Q-table with arbitrary values or zeroes. 
- Then the agent selects an action to take in the current state based on the exploration-exploitation strategy. 
- Initially, the agent explores the environment by taking random actions. 
- Overtime, it gradually shifts towards exploiting the learned information to choose actions that maximize future rewards. 
- After taking an action and observing the resulting state and reward, the agent updates the Q-value for the current state action pair using the following Q-learning update rule:
- ![alt text](image-15.png)
- As the agent learns more about the environment, we decrease the exploration rate over time, allowing the agent to focus more on exploiting the learned information. 
- The Q-Learning process continues until the Q values converge to their optimal values, indicating that the agent has learned the optimal action selection strategy for each state. 
- ![alt text](image-16.png)
- Used in autonomous navigation systems. 
- Main weakness of traditional Q-learning is that it isnt very scalable for more complex environments. 
- It uses a table to store the Q-values, which can become impossible to manage as the state and action spaces get larger for the more complex environments. 
- This is where deep Q-learning becomes advantageous. 

## Deep Q-Learning 
- It leverages deep neural networks to approximate the Q-values instead of using a tabular representation. 
- This approach known as function approximations, allows DQN to handle high dimensional state spaces such as images or raw sensor data, making it suitable for a wide range of complex environments including video games, robotics and autonomous navigation. 
- ![alt text](image-17.png)
- This enables DQN to generalize across similar states and actions, making it more efficient and scalable compared to traditional tabular Q-Learning. 
- ![alt text](image-18.png)
- In pong, we represent the game state as an image frame, where each frame shows the current state of the game, including positions of the paddle and the ball. 
- Agent takes actions based on these image frames, such as moving its paddle up or down, or staying still. 
- ![alt text](image-19.png)
- Agents goal is to learn a Q-function, represented by a deep neural network that estimates the expected future rewards for each possible action given the current state. 
- The neural network takes the image frame of the current state as input and outputs the Q-values for all possible actions. 
- During training the agent interacts with the game environment taking actions based on an epsilon greedy exploration strategy. 
- It collects experience tupes, a tuple with state,action,reward,nextState and uses them to update the Q network parameters using a variant of Q-learning algorithm called deep Q-Learning. 
- ![alt text](image-20.png)
- ![alt text](image-21.png)
- However deep Q-learning is not very good working with raw pixel data from images.
- For this purpose we have Deep convolutional Q-learning. 

## Deep Convolutional Q-Learning
- It is a type of reinforcement learning algorithm that uses deep convolutional neural networks(CNNs) to learn to make decisions in complex environments, particularly those with high dimensional input spaces like images or raw sensory data. 
- ![alt text](image-22.png)
- A CNN is a type of artificial neural network based on how humans see. 
- They are composed of layers of neurons called convolutional layers that apply filters, also called kernels to input data to extract features. 
- ![alt text](image-23.png)
- CNNs are designed to capture spatial hierarchies and patterns in data making them well suited for tasks like image classification and object detection. 
- Here,CNNs learn effective decision making policies in environments with high dimensional input spaces.
- Input is raw sensory data like images or video games. 
- CNN processes the input data, extracting meaningful features that capture spatial relationships and patterns in the environment. 
- Output of the CNN is fed into fully connected layers that estimate the Q- values for all possible actions. 
- e.g environmental monitoring with drones. 
- Goal of these drones is to monitor environmental changes, detect potential threats like wildfires or deforestation and make informed decisions to protect the ecosystem. 
- ![alt text](image-24.png)
- In the input each drone captures images of the forest area using on-board cameras. 
- These images provide visual information about the vegetation, terrain and any potential threats or changes in the environment. 
- Raw image data is fed into a CNN, which processes the images and extracts features relevant to the environmental monitoring such as density and distribution of vegetation, presence of water bodies and signs of deforestation and wildfire. 
- ![alt text](image-25.png)
- In the next step, we do action selection, at which point the output of the CNN is passed to the fully connected layers that estimate the Q-values for different drone locations. Based on these Q-values, the drone takes an action aiming to maximize the effectiveness of monitoring and detection while conserving energy and resources.
-  ![alt text](image-26.png)
-  Positive rewards are given for actions that contribute to effective environmental monitoring such as detecting changes in vegetation health. 
-  Negative rewards or penalties are given for actions that result in inefficiencies or failures, such as colliding with obstacles or running out of battery. 
-  The deep convolutional Q-learning algorithm updates the Q-values based on the observed rewards and transitions between states and actions enabling the drone to learn optimal strategies. 
-  Overtime, the drone learns how to effectively navigate the forest environment. 
-  ![alt text](image-27.png)
  
## Asynchronous Advantage Actor-Critic(A3C)
- It is a type of reinforcement learning algorithm designed for training agents to interact with environments and make decisions in real-time
- 4 parts of A3C 
- ![alt text](image-28.png)
- A3C vs Q-Learning
- A3C is probabilistic vs Q-learning which is deterministic
- ![alt text](image-29.png)
- ![alt text](image-30.png)
- Drones learn from experience and adjust their policies to improve performance over time. 
- For e.g if a drone encounters traffic in a particular area, it may learn to re-route to avoid congestion. 
- If it encounters a high building, it may increase it altitude to skip the obstacle. 
- ![alt text](image-31.png)

## Large Language Models (LLMs)
- Type of AI powered by neural networks 
- ![alt text](image-32.png)
- It is like a smart assistant that can analyze and generate text such as completing sentences, translating languages or even writing articles. 
- LLMs can perform large number of tasks involving natural language processing or NLP. 
- ![alt text](image-33.png)
- ![alt text](image-34.png)
- These models have been trained on massive amounts of text data. 
- ChatGPT is a variant of the GPT model specifically fine tuned for conversational interactions. It is trained on a dataset containing conversational data such as social media conversations, chat logs and customer support interactions. 
- Gives better performance in conversational context. 
- ![alt text](image-36.png)
- LLMs are blackbox systems 
- ![alt text](image-37.png)
- LLM powered chatbots can help in customer service interactions and reduce workload on human agents. 
- LLMs can generate clinical notes and assist with medical documentation. 
- They can convert spoken or handwritten medical records into structured text to improve accuracy and efficiency of healthcare administration. 
- ![alt text](image-38.png)

## Generative AI 
- Type of AI that can be used to generate new, original data or content based on patterns learned from existing data. 
- Unlike traditional AI models that focus on tasks like classification, regression or prediction, generative AI models can create new data samples that are similar to the one in the training data set but are not the exact copies. . 
- These models work by learning the underlying statistical patterns and structures of the data during the training process. 
- Once trained, these models can generate new data samples by sampling from the learned distribution of the data. 
- ![alt text](image-39.png)
- ![alt text](image-40.png)
- The generator generates fake data samples like images from random noise.
- The discriminator tries to distinguish between real data samples, like actual images and fake ones generated by the generator.
- Through adversarial training, the generator learns to generate more realistic data samples, while the discriminator learns to become better at distinguishing between the real and fake data samples.
- GANs are widely used for tasks like image generation, video generation, and data synthesis.

### Variational Encoders
- Variational encoders, or VAEs, are probabilistic generative models that learn a latent representation of the input data.
- They consist of an encoder network that maps input data to a latent space, and a decoder network that generates new data samples from the latent space. VAEs aim to learn the underlying distribution of the input data and generate new data samples by sampling from this distribution.
- They are commonly used for tasks like image generation, anomaly detection, and data compression.

### Autoregressive Models 
- Autoregressive models generate data sequentially, one element at a time based on the probability distribution of the next element given the previous elements.
- Examples of autoregressive models include recurrent neural networks or RNNs, transformer models and language models like GPT.
- Autoregressive models are commonly used for tasks like text generation, language translation, and sequence prediction.
- ![alt text](image-41.png)
- ![alt text](image-42.png)
- ![alt text](image-43.png)


## Computer Vision 
 - Type of AI that focuses on enabling computers to interpret and understand visual information from the real world. 
 - It involves developing algorithms and techniques that allow machines to extract meaningful insights from digital images or videos, similar to how humans perceive and interpret visual stimuli.
 - ![alt text](image-44.png)
 - ![alt text](image-45.png)
 - We can use the Computer Vision program described above to analyze live video feeds from surveillance cameras and identify potential security threats such as unauthorized intruders, suspicious packages, or prohibited items.
 - The program continuously processes video frames in real time, detecting and classifying objects within the scene. 
 - When it detects a suspicious object or activity, it triggers an alert notification to alert security personnel, who can then take appropriate action to investigate and respond to the potential threat.
 - ![alt text](image-46.png)
 - Computer Vision can be used in Radiology.


## Generative AI with ChatGPT 
- ![alt text](image-47.png)
- GPT --> Generative Pre-trained Transformer 
- Trained on large amounts of text from the Internet 
- ![alt text](image-48.png)
- ![alt text](image-49.png)

## AI that solves mazes
- ![alt text](image-50.png)
- PyTorch is a python based library that is used to build machine learning models. 
- Under the hood it uses torch library 
- TorchSharp is the .NET equivalent of PyTorch 
- It is open-source. 
- TorchSharp is a .NET library that provides access to the library that powers PyTorch. It is part of the .NET Foundation.
- The focus is to bind the API surfaced by LibTorch with a particular focus on tensors. The design intent is to stay as close as possible to the Pytorch experience, while still taking advantage of the benefits of the .NET static type system where it makes sense. For example: method overloading is relied on when Pytorch defines multiple valid types for a particular parameter.
- We will create a new project and install the following packages:
```shell 
dotnet add package TorchSharp --version 0.102.4
dotnet add package libtorch-cpu-win-x64 --version 2.2.1.1

```
- This maze has three values zero, one, and two.
- Anywhere you see a zero represents part of a wall.
- The ones are the floor tiles and the two is the goal.
- Then we will start at row 11, column five, which is at the bottom middle row zero, column five at the bottom and the top middle is the goal(where there is a 2)
- But keep in mind that the we can only move anywhere there is a one.
- The zeros in this multi-dimensional array are walls.So we will need to learn to avoid the walls.
- Our maze looks like this 
```c#
int[,] maze1 = {
    //0   1   2   3   4   5   6   7   8   9   10  11
    { 0 , 0 , 0 , 0 , 0 , 2 , 0 , 0 , 0 , 0 , 0 , 0 }, //row 0
    { 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 }, //row 1
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 2
    { 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 3
    { 0 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 4
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 5
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 6
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 7
    { 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 8
    { 0 , 1 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 9
    { 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 10
    { 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0 }  //row 11 (start position is (11, 5))
};

```

### Setting up the Actions and Rewards
- An action is any choice an AI can make 
- ![alt text](image-51.png)
- Our AI has 4 possible action: Left, Right, Up, Down 
- ![alt text](image-52.png)
- This program demonstrates the concept of reinforcement learning by training a model to navigate through a maze. Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.
- The program starts by defining a maze represented as a 2D array. Each cell in the maze has a specific value indicating whether it is a wall, a floor, or the goal. The program then sets up the rewards for each cell in the maze based on these values.
- Next, the program initializes the Q-values tensor, which represents the expected future rewards for taking a particular action in a specific state. The Q-values are initially set to zero.
- The program then defines several helper functions. The determineNextAction function selects the next action to take based on the epsilon-greedy policy, which balances exploration and exploitation. The moveOneSpace function moves the model one space in the maze based on the selected action. The hasHitWallOrEndOfMaze function checks if the model has hit a wall or the end of the maze.
- The main function, trainTheModel, trains the model to navigate through the maze using reinforcement learning. It iterates through a series of episodes, where each episode is a round of training. In each episode, the model starts at the beginning of the maze and takes actions to navigate through the maze until it reaches a wall or the end. At each step, the model selects an action, updates the Q-values using the Q-learning algorithm, and learns from the rewards received.
- After training is complete, the program provides a function called navigateMaze to visualize the path taken by the model in the maze based on the learned Q-values. It prints the moves made by the model, displaying the move count and the coordinates of each move.
- By running this program, you can gain a better understanding of reinforcement learning and how it can be applied to solve problems such as maze navigation.
- In this program, the Q-values for the maze are stored in a tensor called qValues.
- A tensor is a multi-dimensional array that can hold numerical data. In this case, the qValues tensor has dimensions corresponding to the number of rows and columns in the maze, as well as the number of possible actions.
- The Q-values represent the expected future rewards for taking a particular action in a specific state. Each element in the qValues tensor corresponds to a specific cell in the maze and an action. The value at each element represents the Q-value for that cell-action pair.
- The setupQValues function is responsible for setting up the qValues tensor. It initializes all the values to zero, providing a starting point for reinforcement learning algorithms to update and learn from. The dimensions of the qValues tensor are determined by the number of rows and columns in the maze, as well as the number of possible actions.
- Throughout the training process, the Q-values in the qValues tensor are updated based on the rewards received and the expected future rewards. This allows the model to learn and make optimal decisions based on the learned Q-values.


```c#




using TorchSharp;
int[,] maze1 = {
    //0   1   2   3   4   5   6   7   8   9   10  11
    { 0 , 0 , 0 , 0 , 0 , 2 , 0 , 0 , 0 , 0 , 0 , 0 }, //row 0
    { 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 }, //row 1
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 2
    { 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 3
    { 0 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 4
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 5
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 6
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 7
    { 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 8
    { 0 , 1 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 9
    { 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 10
    { 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0 }  //row 11 (start position is (11, 5))
};

const string UP = "up";
const string DOWN = "down";
const string LEFT = "left";
const string RIGHT = "right";

string[] actions = [UP, DOWN, LEFT, RIGHT ];

int[,] rewards;

const int WALL_REWARD_VALUE = -500;
const int FLOOR_REWARD_VALUE = -10;
const int GOAL_REWARD_VALUE = 500;

//This function sets up the rewards for the maze for each cell
void setupRewards(int[,] maze, int wallValue, int floorValue, int goalValue)
{
    int mazeRows = maze.GetLength(0);//0 dimension is the number of rows
    int mazeColumns = maze.GetLength(1); // 1 dimension is the number of columns
    rewards = new int[mazeRows, mazeColumns];

    for (int i = 0; i < mazeRows; i++)
    {
        for (int j = 0; j < mazeColumns; j++)
        {
            switch (maze[i, j])//example value of maze[i,j] is 0, if it is 0, then it is a wall
            {

                case 0:
                    rewards[i, j] = wallValue;
                    break;
                case 1:
                    rewards[i, j] = floorValue;
                    break;
                case 2:
                    rewards[i, j] = goalValue;
                    break;
            }
        }
    }
}

torch.Tensor qValues;
//The setupQValues function is responsible for setting up the Q-values for each cell in the maze.
//Q-values are used in reinforcement learning algorithms to estimate the expected future rewards for taking a particular action in a specific state.
//Overall, the setupQValues function sets up the Q-values tensor with the appropriate dimensions and initializes all the values to zero, providing a starting point for reinforcement learning algorithms to update and learn from.
//A tensor is a fundamental data structure in many programming frameworks, including TorchSharp.
//It is a multi-dimensional array that can hold numerical data. Tensors are similar to arrays or matrices, but they have additional properties and operations that make them suitable for mathematical computations and machine learning algorithms.
//TorchSharp provides a rich set of functions and methods to create, manipulate, and perform computations on tensors. These operations enable efficient numerical computations and make it easier to implement machine learning algorithms.
//In the context of reinforcement learning, tensors can be used to store Q-values, states, actions, and other relevant data
void setupQValues(int[,] maze)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns = maze.GetLength(1);
    qValues = torch.zeros(mazeRows, mazeColumns, actions.Length);
}


// Function will return true if the model has hit a wall or goal and false if the model landed on a floor tile.
bool hasHitWallOrEndOfMaze(int currentRow, int currentColumn, int floorValue)
{
    return rewards[currentRow, currentColumn] != floorValue;
}

//Determine the next action to take 
long determineNextAction(int currentRow, int currentColumn, float epsilon)
{
   Random randon = new Random();
   double randomBetween0And1 = randon.NextDouble();
    // this line of code selects the next action to take in the maze based on a random number and the epsilon value.
    // If the random number is less than epsilon, it chooses the action with the highest Q-value for the current cell. Otherwise, it selects a random action from the available actions.

    long nextAction = randomBetween0And1 < epsilon ? torch.argmax(qValues[currentRow, currentColumn]).item<long>() : randon.Next(actions.Length);
    return nextAction;
}

//Move the model one space in the maze
(int,int) moveOneSpace(int[,] maze,  int currentRow, int currentColumn, long currentAction)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns = maze.GetLength(1);
    int nextRow = currentRow;
    int nextColumn = currentColumn;

    //Move UP, DOWN, LEFT, RIGHT in the maze
    if (actions[currentAction] == UP && currentRow > 0)
    {
        nextRow--;
    } else if (actions[currentAction] == DOWN && currentRow < mazeRows - 1)
    {
        nextRow++;
    }
    else if (actions[currentAction] == LEFT && currentColumn > 0)
    {
        nextColumn--;
    }
    else if (actions[currentAction] == RIGHT && currentColumn < mazeColumns - 1)
    {
        nextColumn++;
    }

    return (nextRow, nextColumn);
}

// The function trainTheModel is responsible for training a model to navigate through a maze using reinforcement learning.
// It takes the maze, floor value, epsilon, discount factor, learning rate, and number of episodes as input parameters.
// The function iterates through a series of episodes, where each episode is a round of training.
// In each episode, the model starts at the beginning of the maze and takes actions to navigate through the maze until it reaches a wall or the end of the maze.
// At each step, the model selects an action based on the epsilon-greedy policy
// and updates the Q-values using the Q-learning algorithm.
// The Q-values represent the expected future rewards for taking a particular action in a specific state.
// The Q-learning algorithm uses the temporal difference error to update the Q-values based on the reward received and the expected future rewards.
// The model learns to navigate through the maze by updating the Q-values through multiple episodes of training.
// After training is complete, the model can use the learned Q-values to make optimal decisions and navigate through the maze efficiently.
// The trainTheModel function is a key component of the reinforcement learning process and enables the model to learn from experience and improve its performance over time.
void trainTheModel(int[,] maze, int floorValue, 
    float epsilon, float discountFactor, float learningRate, float episodes)
{
    for(int episode = 0; episode < episodes; episode++)
    {
        //Each episode is a round of training
        Console.WriteLine("-----Starting episode " + episode + "-----");
        //Set the starting position of the model
        int currentRow = 11;
        int currentColumn = 5;

        while (!hasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
        {
            long currentAction = determineNextAction(currentRow, currentColumn, epsilon);
            int previousRow = currentRow;
            int previousColumn = currentColumn;
            (int,int) nextMove = moveOneSpace(maze, currentRow, currentColumn, currentAction);
            currentRow = nextMove.Item1;
            currentColumn = nextMove.Item2;
            float reward = rewards[currentRow, currentColumn];
            float previousQValue = qValues[previousRow, previousColumn, currentAction].item<float>();
            float temporalDifference = reward + discountFactor * torch.max(qValues[currentRow, currentColumn]).item<float>() - previousQValue;
            float nextQValue = previousQValue + (learningRate * temporalDifference);
            qValues[previousRow, previousColumn, currentAction] = nextQValue;
        }

        Console.WriteLine("-----Finished episode " + episode + "-----");

    }

    Console.WriteLine("Training complete!");
}

//The function navigateMaze is responsible for navigating through a maze based on the learned Q-values. It takes the maze, starting row and column, floor value, and wall value as input parameters.
//The function first initializes an empty list called path to store the coordinates of the model's movement in the maze. If the starting position is a wall or the end of the maze, the function immediately returns an empty list.
//If the starting position is a valid floor tile, the function enters a while loop. Inside the loop, it selects the next action to take based on the epsilon value of 1.0, which means it always chooses the action with the highest Q-value for the current cell.
//The function then moves the model one space in the maze based on the selected action using the moveOneSpace function. It updates the current row and column accordingly.
//If the next position is not a wall, it adds the current position to the path list. Otherwise, it continues to the next iteration of the loop.
//Once the model reaches a wall or the end of the maze, the function exits the loop. It then iterates through the path list and prints the moves made by the model, displaying the move count and the coordinates of each move.
//Finally, the function returns the path list, which contains the coordinates of the model's movement in the maze.
//This function allows you to visualize the path taken by the model in the maze based on the learned Q-values, providing insights into the model's decision-making process and its ability to navigate through the maze efficiently.
List<int[]> navigateMaze(int[,] maze, int startRow, int startColumn, int floorValue, int wallValue)
{
    List<int[]> path = new List<int[]>();
    if (hasHitWallOrEndOfMaze(startRow, startColumn, floorValue))
    {
        return [];
    }
    else
    {
        int currentRow = startRow;
        int currentColumn = startColumn;
        path = [[currentRow, currentColumn]];
        while (!hasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
        {
            int nextAction = (int) determineNextAction(currentRow, currentColumn, 1.0f);
            (int, int) nextMove = moveOneSpace(maze, currentRow, currentColumn, nextAction);
            currentRow = nextMove.Item1;
            currentColumn = nextMove.Item2;
            if(rewards[currentRow, currentColumn] != wallValue)
            {
                path.Add([currentRow, currentColumn]);
            }
            else
            {
                continue;
            }
        }
    }

    int moveCount = 1;
    for(int i = 0; i < path.Count; i++)
    {
        Console.WriteLine("Move " + moveCount + ": (");
        foreach(int element in path[i])
        {
            Console.WriteLine(" "+element);
        }
        Console.WriteLine(" )");
        Console.WriteLine();
        moveCount++;
    }

    return path;
}

const float EPSILON = 0.95f;
const float DISCOUNT_FACTOR = 0.8f;
const float LEARNING_RATE = 0.9f;
const int EPISODES = 1500;
const int START_ROW = 11;
const int START_COLUMN = 5;

setupRewards(maze1, WALL_REWARD_VALUE, FLOOR_REWARD_VALUE, GOAL_REWARD_VALUE);
setupQValues(maze1);
trainTheModel(maze1, FLOOR_REWARD_VALUE, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE, EPISODES);
navigateMaze(maze1, START_ROW, START_COLUMN, FLOOR_REWARD_VALUE, WALL_REWARD_VALUE);


```
### Result is as follows: 
- ![alt text](image-53.png)

## Neural Networks
- It is basically as a digital brain
- It consists of neurons which receive an input and produces an output. 
- This neuron applies a mathematical function called an activation function on the inputs to produce the output 
- ![alt text](image-54.png)
- ![alt text](image-55.png)
- Neurons are organized into layers 
- ![alt text](image-56.png)
- Imagine we want to build a system that recognizes handwritten digits 
- We feed a neural network images of handwritten digits along with their corresponding labels 
- The neural network learns to extract features from the image like edge, curves and patterns 
- In the output later, the network assigns probability to each digit. 
- For instance, if the network predicts a high probability for digit five, it means it's confident that the input image represents the digit five.
- ![alt text](image-57.png)

### Neural Network Architecture  
- ![alt text](image-58.png)
- Main components of neural network architecture are:
- Neurons 
- Layers 
- Weighted Connections 
- ![alt text](image-59.png)
- Neurons are organized into layers 
- Each layer connects to the next. 
- The number of neurons in each layer can vary 
- Weights determine strength of influence on the neuron 
- ![alt text](image-60.png)
#### Feed-Forward Neural Network 
- ![alt text](image-61.png)
- Used for tasks like classification and regression. 
- Lets say we have an image of an fruit, this neural network can help recognize it based on the input features.
- To train this network, we need labelled data.
- Once trained, this neural network can now take in new input data like the color and size of a fruit, and predict whether it's an apple or an orange.

#### Convolutional Neural Network 
- ![alt text](image-62.png)
- Convolutional neural networks, or CNNs, are specialized for processing grid like data such as images.
- They consist of convolutional layers that extract features from input images, followed by pooling layers for downsampling and fully connected layers for classification.
- Imagine you want to build a system that can recognize whether a picture contains a cat or a dog.
A CNN can help you achieve this.
- In a CNN, the first layer is typically a convolutional layer. This layer consists of tiny filters, also called kernels or feature detectors. Each filter slides across the input image pixel by pixel, and performs a mathematical operation called convolution.
- As the filters slide over the image, they detect patterns such as edges, textures, and shapes.
- These patterns are the building blocks of objects in the image.
- For example, one filter might detect horizontal edges while another detects vertical edges.
- After convolution, the result passes through an activation function such as ReLU, which stands for
rectified linear unit. This function introduces non-linearity to the network, allowing it to learn complex relationships between features.
- Next, the output from the convolutional layer is typically passed through a pooling layer.
- Pooling helps reduce the spatial dimensions of the input, making the network more computationally efficient and less sensitive to small changes in the input after several convolutional and pooling layers.
- The extracted features are flattened into a vector and passed to one or more fully connected layers.
- These layers perform high level reasoning and decision making based on the learned features.
- Finally, the output layer of the CNN contains neurons corresponding to the possible classes like cat
or dog.
- To train a CNN, we need a dataset of labelled images like images of dogs,cats etc. 
- Once trained, if fed an image, the neural network can predict whether it is a cat or dog. 

#### Recurrent Neural Network 
- ![alt text](image-63.png)
- Recurrent neural networks, or RNNs, are designed to handle sequential data where the order of inputs
matters.
- They have connections that loop back on themselves, allowing them to capture temporal dependencies
in the data.
- RNNs are widely used in tasks like speech recognition, language modeling, and time series prediction.
- Imagine you want to build a system that generates text, one character at a time, based on a given
input. An RNN can help you achieve this by remembering the context of the previous characters as it generates the next one.
- At its core, RNN consists of a chain of repeating neural network modules or cells.
- Each cell takes in two inputs the current input, like the character and the hidden state from the previous cell.
- The hidden state acts as the memory of the network, capturing information about past inputs.
- As the input sequence is fed into the RNN one element at a time, the hidden state is updated at each
step based on the current input and the previous hidden state.
- This process allows the network to maintain context and capture dependencies between sequential elements.
- One variant is LSTM(Long short term memory) network.
- Suppose we want to generate a sentence based on a given starting word.
- We input the starting word into the RNN and let it generate the next word.
- Then we input the generated word back into the RNN along with the previous hidden state, and repeat
this process until we reach the desired length of the sentence.
- During training, the RNN learns to predict the next element in the sequence based on the previous elements.
- This is achieved by adjusting the parameters like weights and biases of the network using backpropagation through time or bptt, a variant of the backpropagation algorithm that takes into account the sequential nature of the data.
- ![alt text](image-64.png)
- LSTM is much better for speech recognition, time-series prediction
- It builds a context of the input text.

### Creating a Neural Network 
- ![alt text](image-65.png)
- We must understand the sigmoid function 
- ![alt text](image-66.png)
- Think of a sigmoid function that takes in any input value and gives us a value between 0 and 1. 
- It can convert any value big or small and give us a value between 0 and 1 which is easy to work with. 
- ![alt text](image-67.png)
- derivative of any function tells us how much the function is changing at any given point of time. 
- For the sigmoid function, its derivative tells you the rate at which the output is changing concerning
the input.
- When you take the derivative of the sigmoid function, you get another function that also depends on
the input value x.
- This derivative function helps us adjust the weights in our neural network during the training process.
- The derivative function is helpful because it tells us how quickly the output of the sigmoid function
is changing concerning the input x.
- We can use this information to adjust the parameters of our neural network during the training process, which helps our network learn and improve over time.

### Coding the Train Function
- The Train method in the NeuralNetwork class is responsible for training the neural network using a process called backpropagation. Let's break down the steps in the method:
1.	Iteration Loop: The method starts with a loop that iterates numberOfIterations times. This allows the neural network to update its weights multiple times to improve its performance.
2.	Forward Pass: Inside the loop, the Think method is called to perform a forward pass through the neural network. The Think method takes the trainingInputs as input and returns the output of the neural network.
3.	Error Calculation: The next step is to calculate the error between the expected output (trainingOutputs) and the actual output of the neural network. This is done by calling the PerformOperation method with the OPERATION.SUBTRACT operation. The PerformOperation method subtracts the trainingOutputs from the output obtained from the forward pass.
4.	Adjustment Calculation: The adjustment to the weights is calculated using the backpropagation algorithm. The backpropagation algorithm calculates the gradient of the error with respect to the weights and adjusts the weights accordingly. The adjustment is calculated by performing a series of matrix operations:
•	The Activate method is called with the output and true as arguments to calculate the derivative of the sigmoid activation function. The derivative is used to scale the error based on the slope of the activation function.
•	The PerformOperation method is called with the OPERATION.MULTIPLY operation to multiply the error by the derivative of the sigmoid function.
•	The DotProduct method is called with the transposed trainingInputs and the result of the previous operation to calculate the dot product between the transposed inputs and the error multiplied by the derivative.
•	Finally, the PerformOperation method is called with the OPERATION.ADD operation to add the adjustment to the current weights.
5.	Update Weights: The updated weights are assigned back to the weights variable of the neural network.
By repeating this process for the specified number of iterations, the neural network gradually adjusts its weights to minimize the error and improve its performance in producing accurate outputs for the given inputs.

```c#
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


```

### DotProduct Function
- The DotProduct function calculates the dot product of two matrices. The dot product is a mathematical operation that takes two matrices and produces a new matrix by multiplying corresponding elements and summing the results.
Here's an example to illustrate how the DotProduct function works:
- Suppose we have two matrices:
matrix1 = [[1, 2],
           [3, 4]]

matrix2 = [[5, 6],
           [7, 8]]

- The dot product of matrix1 and matrix2 can be calculated as follows:
- result = [[1*5 + 2*7, 1*6 + 2*8],
          [3*5 + 4*7, 3*6 + 4*8]]
- Simplifying the calculation, we get:
- result = [[19, 22],
          [43, 50]]
- So, the resulting matrix result will have the same number of rows as matrix1 and the same number of columns as matrix2. Each element in the resulting matrix is obtained by multiplying the corresponding elements from matrix1 and matrix2 and summing them.
- In the provided code, the DotProduct function takes two matrices, matrix1 and matrix2, as input. 
- It iterates over the rows of matrix1 and the columns of matrix2, calculating the dot product for each element in the resulting matrix. The resulting matrix is then returned as the output of the function.
```c#
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
```


### Think method in Neural Method class
- The Think method in the NeuralNetwork class is responsible for performing a feedforward operation in the neural network. It takes a 2D array of inputs as a parameter and returns a 2D array of the resulting outputs.
- We call the Think method on the neuralNetwork object, passing in the inputs array. The Think method performs a feedforward operation by first calculating the dot product of the inputs and the weights of the neural network. The dot product is calculated using the DotProduct method.
The resulting dot product is then passed to the Activate method, which applies the sigmoid activation function to each element in the dot product matrix. The sigmoid function transforms the values to a range between 0 and 1, representing the output of the neural network.
Finally, the Think method returns the resulting outputs as a 2D array, which we can then use for further processing or analysis.
```c#
 public double[,] Think(double[,] inputs)
{
    return Activate(DotProduct(inputs, weights), false);
}

```

### Transpose Function in Neural Network
- Transposing a 2D array is like flipping it over its diagonal 
- Rows become columns and columns become rows
- ![alt text](image-68.png)
```c#
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
```

### Testing the code 
- The method **Train** in the NeuralNetwork class is responsible for training the neural network using the provided training data. Let's break down the steps performed in this method:
1.	The method takes three parameters: trainingInputs, trainingOutputs, and numberOfIterations.
•	trainingInputs is a 2D array that represents the input data for the neural network. Each row in the array represents a set of input values.
•	trainingOutputs is a 2D array that represents the expected output data for the corresponding input data. Each row in the array represents the expected output for the corresponding input row.
•	numberOfIterations specifies the number of iterations or epochs for which the training process will be performed.
2.	The method starts a loop that iterates numberOfIterations times. This loop represents the training process.
3.	Inside the loop, the **Think** method is called to obtain the output of the neural network for the current set of trainingInputs. The Think method performs a feedforward operation, applying the activation function to the dot product of the trainingInputs and the current weights of the neural network.
4.	The **difference** between the **obtained** output and the **expected** output, called the **error**, is calculated by performing the subtraction operation between trainingOutputs and the obtained output.
5.	The **Activate** method is called to apply the **derivative** of the sigmoid activation function to the obtained output. This is done by passing the obtained output and true as the isDerivative parameter. The Activate method returns the derivative of the sigmoid function if isDerivative is true, otherwise it returns the sigmoid output.
6.	The Transpose method is called to transpose the trainingInputs matrix. Transposing a matrix means converting its rows into columns and its columns into rows. This is useful for performing matrix multiplication.
7.	The DotProduct method is called to perform the dot product between the transposed trainingInputs and the result of the element-wise multiplication of error and the derivative of the obtained output. The dot product is a mathematical operation that calculates the sum of the products of corresponding elements in two matrices.
8.	The resulting dot product, called **adjustment**, represents the **adjustment** to be made to the current weights of the neural network.
9.	The **PerformOperation** method is called to *add the adjustment* to the current weights. The PerformOperation method performs element-wise addition between the two matrices.
10.	The updated weights are assigned to the weights variable of the neural network.
11.	The loop continues until the specified number of iterations is reached.
In summary, the Train method trains the neural network by iteratively adjusting the weights based on the error between the expected output and the obtained output. This process helps the neural network learn and improve its ability to make accurate predictions.

```c#
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

```

## Final Code:
```c#
 
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


```

## Real world applications of Neural Networks
- ![alt text](image-69.png)
- ![alt text](image-70.png)
- Neural networks are used in healthcare like radiology.
- Also used for natural language processing
- Can be used in self-driving cars
- In finance industry, can be used to detect frauds
- Also allow robots to view their environment

## Why we need sigmoid function?
- A neural network is like a brain-inspired system in a computer that learns to make decisions or predictions. It’s made of layers of "neurons" (small units) that process numbers and pass them along. These neurons take inputs (like numbers), do some math, and produce an output.

- The problem is: the raw output of a neuron (after doing some math) can be any number, like -5, 10, or 1000. But in many cases, we need the output to be in a specific range (like 0 to 1) to make sense for things like probabilities or decisions. That’s where the sigmoid function comes in!
- The sigmoid function is a mathematical formula that "squashes" any number into a range between 0 and 1. It’s shaped like an "S" curve and looks like this:
- ![alt text](image-71.png)
- x is the input number (can be anything: positive, negative, big, small).
- e is a special math constant (~2.718).
- The output is always between 0 and 1.
- How it Works:
- If x is a big positive number (e.g., 10), sigmoid output is close to 1.
- If x is a big negative number (e.g., -10), sigmoid output is close to 0.
- If x is 0, sigmoid output is 0.5.
- This makes it perfect for situations where we want to interpret the output as a probability (like "how likely is this to be true?").
- Neural networks often need to:
- Make Decisions: For example, "Is this a cat in the picture? Yes or No."
- Learn: Adjust their internal math based on how wrong their predictions are.
- The sigmoid function helps with both:
- Squashing Outputs: It turns raw numbers into a 0-to-1 range, which can represent probabilities or "confidence levels."
- Smooth Learning: Its "S" shape is smooth and gradual, which helps the network learn by giving it a way to tweak its predictions little by little (using something called "gradient descent").
- Without sigmoid (or similar functions), the neuron outputs could be all over the place, making it hard to interpret or train the network.
- Imagine you’re building a tiny neural network to decide: "Is this a sunny day?" based on two inputs:
- Temperature (e.g., 80°F).
- Cloud cover (e.g., 20% cloudy).
- Step 1: Raw Math in a Neuron
- The neuron takes these inputs, multiplies them by "weights" (importance factors), adds them up, and gets a number. Let’s say:

- Temperature (80) × Weight (0.1) = 8
- Cloud cover (20) × Weight (-0.2) = -4
- Total = 8 + (-4) = 4
- So, the neuron’s raw output is 4. What does "4" mean? It’s not clear—it’s just a number!

- Step 2: Apply Sigmoid
- Now, plug 4 into the sigmoid function:
- ![alt text](image-72.png)
- The output is 0.982 (close to 1). This could mean "98.2% chance it’s a sunny day!" That’s way easier to understand than "4."


## Machine Learning with .NET
- Open source ML framework developed by Microsoft. 
- Enables developers to build and incorporate ML models into .NET Applications 
- It provides a set of tools, libraries, and APIs that streamline the process of creating, training,and deploying machine learning models, all within the familiar environment of the dotnet ecosystem.
- ![alt text](image-73.png)
### Workflow of ML.NET 
- ![alt text](image-74.png)
### Applications of ML.NET 
- Ml.net supports NLP tasks such as sentiment analysis, named entity recognition and text classification, enabling developers to analyze and extract insights from textual data.
- Ml.net includes components for computer vision tasks such as image classification, object detection,
and image segmentation, allowing developers to build applications that understand and interpret visual
information.
- ![alt text](image-75.png)
- Add the following nuget package for incorporating ML.NET in our application 
```shell
dotnet add package Microsoft.ML
dotnet add package Microsoft.ML.FastTree
```
### We will create a program using ML.NET which will perform regression on a given dataset(housing-data) and predict house prices 
- The code loads housing data, creates a machine learning pipeline, trains a regression model, makes predictions, evaluates the model's performance, and prints the evaluation metrics.

- MLContext mlContext = new MLContext();: This line creates an instance of the MLContext class, which is the main entry point for ML.NET functionality. It provides the environment for creating and executing machine learning workflows.
  
- IDataView data = mlContext.Data.LoadFromTextFile<HousingData>("housing-data.csv", separatorChar: ',');: This line loads the housing data from a text file called "housing-data.csv" and converts it into an IDataView object. The LoadFromTextFile method is used to read the data from the file, and the generic type parameter HousingData specifies the class that represents the structure of the data.
  
- string[] featureColumns = { "SquareFeet", "Bedrooms" };: This line defines an array of strings that represents the names of the columns in the housing data that will be used as features for the regression task. In this case, the "SquareFeet" and "Bedrooms" columns are selected as features.
  
- string labelColumn = "Price";: This line defines a string variable that represents the name of the column in the housing data that will be used as the label for the regression task. In this case, the "Price" column is selected as the label.
  
- var pipeline = mlContext.Transforms.Concatenate("Features", featureColumns).Append(mlContext.Regression.Trainers.FastTree(labelColumnName: labelColumn));: This line creates a machine learning pipeline. The Concatenate method is used to combine the feature columns into a single column called "Features". The Append method is used to add a regression trainer to the pipeline. In this case, the FastTree trainer is used, which is a decision tree-based regression algorithm.
  
- var model = pipeline.Fit(data);: This line trains the machine learning model by fitting the pipeline to the loaded data. The Fit method takes the data as input and returns a trained model.
  
- var prediction = model.Transform(data);: This line uses the trained model to make predictions on the same data that was used for training. The Transform method takes the data as input and returns a new IDataView object containing the predicted values.
  
- var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: labelColumn);: This line evaluates the performance of the model by comparing the predicted values with the actual values in the data. The Evaluate method takes the predicted values and the label column name as input and returns a set of regression evaluation metrics.
  
- Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");: This line prints the mean absolute error metric, which measures the average absolute difference between the predicted and actual values.
  
- Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");: This line prints the root mean squared error metric, which measures the square root of the average squared difference between the predicted and actual values.

```c#
 using Microsoft.ML;
using Microsoft.ML.Data;


MLContext mlContext = new MLContext();
IDataView data = mlContext.Data.LoadFromTextFile<HousingData>("housing-data.csv", separatorChar: ',');
string[] featureColumns = { "SquareFeet", "Bedrooms" };
string labelColumn = "Price";

// Define the training pipeline
var pipeline = mlContext.Transforms.Concatenate("Features", featureColumns)
    .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: labelColumn));

// Train the model
var model = pipeline.Fit(data);

// Make predictions
var prediction = model.Transform(data);

// Evaluate the model
var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: labelColumn);

// Print the evaluation metrics
Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");


// Define a class to hold the housing data
public class HousingData
{
    [LoadColumn(0)]
    public float SquareFeet { get; set; }

    [LoadColumn(1)]
    public float Bedrooms { get; set; }

    [LoadColumn(2)]
    public float Price { get; set; }
}

// Define a class to hold the housing prediction
public class HousingPrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}

```
### Data Preparation and Loading in ML.NET
- ML.NET provides various methods for loading data, including loading from text files, databases and in-memory collections.
- If the data is already available in memory as a collection(e.g a list or an array), we can use the LoadFromEnumerable() method to load it directly into ML.NET. This is useful when your data is relatively small and can fit into memory. 
- If the data is stored in a database, we can use the LoadFromDatabase() to load it into ML.NET. 
- This method allows us to specify a database connection string and SQL query to retrieve the data.
- If our text file contains a header row with column names, we can use the LoadFromTextFileWithHeader method to load the data while automatically inferring the column types. 
- If our data is stored in JSON format, we can use the LoadFromJson() method to load it into ML.NET. This method allows us to specify the JSON file path and the data schema. 
- If our data is in Apache Parquet format, we can use the LoadFromParquetFile() method to load it into ML.NET. Parquet is a columnar storage format that is efficient for large datasets. 
- ![alt text](image-76.png)
- Ml.net provides methods for cleaning data such as replace missing values, filter rows by missing values, and remove duplicates.
- Ml.net provides transformers for feature engineering tasks such as concatenate, normalize, and one
hot encoding.
- Split your data into training and testing sets to evaluate your model's Performance. Ml.net provides methods for splitting data such as train test, split and cross validation.
- Ml.net provides transformers for normalization such as normalize min max and normalize mean variance.
- Pipeline data preparation steps create a data preparation pipeline to automate and streamline the data preparation process.

### Feature Engineering in ML.NET 
- Feature engineering is the process of creating new features from existing ones or transforming existing features to improve the performance of machine learning models. 
- It is like giving our model the right tools to make accurate predictions.
- Plays crucial role in success of ML models. 
- By engineering meaningful features, we can capture more relevant information from the data, reduce
noise, and improve the model's ability to generalize to unseen data.
- Good feature engineering can make or break a machine learning model.
- In machine learning, features are the pieces of information (or input variables) that a model uses to make predictions. For example, if you’re predicting whether someone will buy a product, features might include their age, income, or location. Feature engineering is the process of preparing and improving these features so that your machine learning model can learn from them more effectively. It’s like organizing and polishing raw data to help the model do its job better.
- In ML.NET, feature engineering involves transforming raw data into a format that the model can understand, creating new useful features, and selecting the most relevant ones. ML.NET makes this easier by providing tools called transformers, which you can chain together in a pipeline to preprocess your data before training a model.
- Transforming existing features: Turning data into a usable form (e.g., converting text or categories into numbers).
- Creating new features: Combining or modifying existing data to make it more meaningful.
- Selecting features: Picking the most important ones to avoid confusing the model with irrelevant details.
- Imagine you have a dataset about houses with these columns:
- Number of bedrooms (e.g., 3)
- Size in square feet (e.g., 1500)
- Age of the house (e.g., 10 years)
- Price (e.g., $300,000)
- Your goal is to predict the price of a house based on the other columns.
- Here’s how feature engineering comes into play:
- Using Existing Features
You can use "number of bedrooms," "size," and "age" directly as features. These are already numbers, so the model can work with them. But we can do more to improve things!
- Creating a New Feature
What if the price depends not just on size, but on how spacious each bedroom feels? You could create a new feature called size per bedroom by dividing "size" by "number of bedrooms" (e.g., 1500 ÷ 3 = 500 square feet per bedroom). This new feature might give the model extra insight into what makes a house valuable.
- Scaling Features
Notice that "size" (1500) is much bigger than "age" (10) or "bedrooms" (3). Some machine learning algorithms get confused when features are on different scales. In ML.NET, you can use a transformer like NormalizeMinMax to adjust all features to a range like 0 to 1, making them easier for the model to compare.

```c#
 var pipeline = mlContext.Transforms.NormalizeMinMax("SizeNormalized", "Size")
    .Append(mlContext.Transforms.NormalizeMinMax("AgeNormalized", "Age"))
    .Append(mlContext.Transforms.Concatenate("Features", "SizeNormalized", "AgeNormalized", "Bedrooms"));

```
#### Feature Engineering Techniques
- Normalization 
- ![alt text](image-77.png)
- ![alt text](image-78.png)
- OneHotEncoding is a technique used in machine learning to convert categorical data—information that represents categories or labels, such as colors (red, blue, green), sizes (small, medium, large), or countries (USA, Canada, Mexico)—into a numerical format that machine learning algorithms can understand. Since most machine learning models work with numbers rather than text or labels, OneHotEncoding provides a way to represent categories in a way that’s suitable for these algorithms.
- Machine learning algorithms often assume that numerical data has a natural order or magnitude. For example, if you assign numbers to categories like this:

Red = 1
Blue = 2
Green = 3
The model might incorrectly assume that green (3) is "greater" than red (1), implying a ranking or relationship that doesn’t exist—colors are just different, not ordered. OneHotEncoding solves this by representing each category as a binary vector, ensuring that categories are treated independently without suggesting any hierarchy.
- ![alt text](image-79.png)

```c#
 using Microsoft.ML;
using Microsoft.ML.Data;


MLContext mlContext = new MLContext();
IDataView data = mlContext.Data.LoadFromTextFile<HousingData>("housing-data.csv", separatorChar: ',');

// Define the data preparation pipeline
// Convert the SquareFeet column to a Single type
// Normalize the SquareFeet column
// Concatenate the SquareFeet and Bedrooms columns into a Features column
// One-hot encode the Neighborhood column

var dataPipeline = 
    mlContext.Transforms.Conversion.ConvertType("SquareFeet", outputKind: DataKind.Single)
    .Append(mlContext.Transforms.NormalizeMinMax("SquareFeet"))
    .Append(mlContext.Transforms.Concatenate("Features", "SquareFeet", "Bedrooms"))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"));

// Fit and transform the data
var transformedData = dataPipeline.Fit(data).Transform(data);

// Print the transformed data
var transformedDataEnumerable = mlContext.Data.CreateEnumerable<TransformedHousingData>(transformedData, reuseRowObject: false).ToList();

foreach (var item in transformedDataEnumerable)
{
    Console.WriteLine($"SquareFeet: {item.SquareFeet}," +
        $" Bedrooms: {item.Bedrooms}, " +
        $"Price: {item.Price}, " +
        $"Features: [{string.Join(", ", item.Features)}], " +
        $"Neighborhood: [{string.Join(", ", item.Neighborhood)}]");
}

```

## Model Selection and Evaluation in ML.NET 
- The model selection process involves choosing the best algorithm and hyper-parameters for our machine learning task. 
- ![alt text](image-80.png)
- ![alt text](image-81.png)
- ![alt text](image-82.png)

```c#
 using Microsoft.ML;
using Microsoft.ML.Data;


static void EvaluateMetrics(string modelName, BinaryClassificationMetrics metrics)
{
    Console.WriteLine($"{modelName} - Accuracy:{metrics.Accuracy:0.##}");
    Console.WriteLine($"{modelName} - AUC:{metrics.AreaUnderRocCurve:0.##}");
}

var context = new MLContext();
var data = context.Data.LoadFromTextFile<DataPoint>("data.csv", separatorChar: ',', hasHeader:true);

// Split the data into training and test sets
var trainTestSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

// Train the model
// Define the pipeline
// Concatenate the features into a single column
// Append the logistic regression trainer
// The label column is the "Label" column
// The maximum number of iterations is 100
// context.Transforms.Concatenate("Features", "Feature1", "Feature2") is a transformation step that concatenates multiple input features into a single column.
// In this example, we are concatenating two features, "Feature1" and "Feature2", into a new column called "Features".
// This transformation is useful when you want to combine multiple features into a single input for the model.
// Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName:"Label", maximumNumberOfIterations: 100)) is the trainer step that appends a logistic regression trainer to the pipeline. The trainer is responsible for training the model using the transformed data. In this example, we are using the SdcaLogisticRegression trainer, which is a type of logistic regression algorithm.
// We specify the label column name as "Label" and set the maximum number of iterations to 100.
var logisticRegressionPipeline = context.Transforms.Concatenate("Features", "Feature1", "Feature2")
    .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName:"Label", maximumNumberOfIterations: 100));


var fastTreePipeline = context.Transforms.Concatenate("Features", "Feature1", "Feature2")
    .Append(context.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", numberOfLeaves: 50, numberOfTrees:100));

Console.WriteLine("Training Logistic Regression model...");
var logisticRegressionModel = logisticRegressionPipeline.Fit(trainTestSplit.TrainSet);

Console.WriteLine("Training FastTree model...");
var fastTreeModel = fastTreePipeline.Fit(trainTestSplit.TrainSet);

// Evaluate the models
Console.WriteLine("Evaluating the Logistic Regression Model...");
var logisticRegressionPredictions = logisticRegressionModel.Transform(trainTestSplit.TestSet);
var logisticRegressionMetrics = context.BinaryClassification.Evaluate(logisticRegressionPredictions, "Label");
EvaluateMetrics("Logistic Regression", logisticRegressionMetrics);

Console.WriteLine("Evaluating the FastTree Model...");
var fastTreePredictions = fastTreeModel.Transform(trainTestSplit.TestSet);
var fastTreeMetrics = context.BinaryClassification.Evaluate(fastTreePredictions, "Label");
EvaluateMetrics("FastTree", fastTreeMetrics);

if(logisticRegressionMetrics.Accuracy > fastTreeMetrics.Accuracy)
{
    Console.WriteLine("Logistic Regression Model is the better model");
} else if(logisticRegressionMetrics.Accuracy < fastTreeMetrics.Accuracy)
{
    Console.WriteLine("FastTree Model is the better model");
}
else
{
    Console.WriteLine("Both models are equally good");
}

public class DataPoint
{
    [LoadColumn(0)]

    public float Feature1 { get; set; }
    [LoadColumn(1)]
    public float Feature2 { get; set; }

    [LoadColumn(2)]
    public bool Label { get; set; }
}

public class Prediction
{
    [ColumnName("Score")]
    public float Score { get; set; }

    [ColumnName("Probability")]
    public float Probability { get; set; }
}



```

## Training and Tuning the models in ML.NET 
- ![alt text](image-83.png)
- ![alt text](image-84.png)
```c#
 var logisticRegressionPipeline = context.Transforms.Concatenate("Features", "Feature1", "Feature2")
    .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName:"Label", maximumNumberOfIterations: 100));


var fastTreePipeline = context.Transforms.Concatenate("Features", "Feature1", "Feature2")
    .Append(context.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", numberOfLeaves: 50, numberOfTrees:100));

Console.WriteLine("Training Logistic Regression model...");
var logisticRegressionModel = logisticRegressionPipeline.Fit(trainTestSplit.TrainSet);

Console.WriteLine("Training FastTree model...");
var fastTreeModel = fastTreePipeline.Fit(trainTestSplit.TrainSet);
```

### Model Deployment and Integration with ML.NET 
- Model deployment involves making the trained machine learning model available for use in Production Environments.
- ![alt text](image-85.png)
- Model serialization.
- The trained Ml.net model needs to be serialized into a format that can be easily loaded and used by
production systems.
- Ml.net supports model serialization to different formats, including Onnx, Open Neural Network Exchange, and the native Ml.net format model hosting.
- Once serialized, the model needs to be hosted within an application or service where it can receive
input data, make predictions, and return results.
- This can be achieved by embedding the model within a web service, a serverless function, or a dedicated microservice.
- Scalability and performance.
- When deploying models into production, scalability and performance are crucial factors to consider.
- Ml.net models can be deployed to scalable cloud platforms like Azure, where they can benefit from auto scaling and high performance infrastructure.
- ![alt text](image-86.png)
- ![alt text](image-87.png)
- Models can also be packaged into docker containers.
- Ml.net supports building and deploying complex model pipelines that include pre-processing, feature
engineering, and model inference steps.
- These pipelines can be integrated into existing data processing pipelines or workflows to automate the end to end machine learning process.(MLOps)
- Implement logging and monitoring mechanisms to track model performance, detect anomalies and troubleshoot issues in production.
- Maintain version control for deployed models to track changes.

## Creating a Classification AI using ML.NET
- Classification in machine learning is about teaching computers to sort data into different categories based on patterns they learn from past examples. 
- ![alt text](image-88.png)
- Fruits are our data points and type of fruit is the category or label 
- ![alt text](image-89.png)
- We can use classification AI to determine if email is spam or not 
- ![alt text](image-90.png)
- ![alt text](image-91.png)
- ![alt text](image-92.png)
### Training the Classification AI model to a csv file containing movie reviews 
- First we will clean the training data 
- The following code will replace all quotes(') with empty text so that our model is not confused
```c#
 //Create a new MLContext instance
MLContext mLContext = new MLContext();
Load the data from the file into an IDataView object
string dataPath = "train.csv";
string text = File.ReadAllText(dataPath);
using (StreamReader streamReader = new StreamReader(dataPath))
{
   text = text.Replace("\'", "");
} 

File.WriteAllText(dataPath, text);

```
- Next we will start training the model 
```c#
//Create a new MLContext instance
MLContext mLContext = new MLContext();
//Load the data from the file into an IDataView object
string dataPath = "train.csv";
//Load training data into a dataView
IDataView dataView = mLContext.Data.LoadFromTextFile<MovieReview>(dataPath, hasHeader: true, allowQuoting: true, separatorChar: ',');

//Create a pipeline which featurizes text and use Logistic Regression algorithm
//The pipeline starts with the Transforms.Text.FeaturizeText method.
//This method is used to convert the text data into numerical features that can be used by the machine learning algorithm.
//It takes two parameters: the name of the output column ("Features") and the name of the input column ("text").
//The Append method is then called on the pipeline to add another component to the sequence.
//In this case, it appends the BinaryClassification.Trainers.SdcaLogisticRegression method, which represents the chosen machine learning algorithm.
//This algorithm is a binary logistic regression model trained using the Stochastic Dual Coordinate Ascent(SDCA) optimization algorithm.
//It takes two parameters: the name of the label column("Label") and the name of the feature column("Features").
//in summary, this line of code creates a pipeline that first converts the text data into numerical features and then applies a binary logistic regression model to train the data.
var pipeline = mLContext.Transforms.Text.FeaturizeText("Features", "text")
    .Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));


//In machine learning, a pipeline is a sequence of data processing components, called transformers and estimators, that are applied in a specific order to transform the data and train a model.
//The Fit method is used to train the model by fitting the pipeline to the data.
//In this case, the pipeline variable represents the sequence of transformations and the chosen machine learning algorithm.
//The Fit method takes the dataView as input and trains the model by applying the transformations and the chosen algorithm to the data.
var model = pipeline.Fit(dataView);

//The Transform method is used to apply the trained model to new data and generate predictions.
var predictions = model.Transform(dataView);

//The Evaluate method is used to evaluate the model's performance on the test data.
var metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");
Console.WriteLine($"Accuracy: {metrics.Accuracy}");
Console.WriteLine($"Accuracy: {metrics.PositivePrecision}");
Console.WriteLine($"Accuracy: {metrics.PositiveRecall}");
Console.WriteLine($"Accuracy: {metrics.F1Score}");

//Save the model to a file
mLContext.Model.Save(model, dataView.Schema, "sentiment_model.zip");


```

### Evaluating and Testing the Model
- We will load the sample test file into the project. Then we will load the zipped model first. 
- Then we will load the test data into the DataView
- We will then create a  prediction engine using the MLContext's Model.CreatePredictionEngine method. - The prediction engine is used to make predictions on new data using a trained ML.NET model.
- In this specific case, the TextData class is used as the input type for the prediction engine, and the SentimentPrediction class is used as the output type. The model variable represents the trained ML.NET model that was loaded from a file.
- Once the prediction engine is created, you can use it to make predictions by calling the Predict method and passing in an instance of the TextData class. 
- The prediction engine will apply the trained model to the input data and generate a prediction based on the model's learned patterns.
- For example, in the code snippet you provided, the prediction engine is used in a loop to make predictions on a list of TextData instances. The predicted sentiment (positive or negative) is then printed to the console.
```c#
  string modelPath = "sentiment_model.zip";
 string testDataPath = "movieReviewsTesting.csv";
 var mlContext = new MLContext();
 ITransformer model;
 using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
 {
     //Load the model from the file
     model = mlContext.Model.Load(stream, out var modelInputSchema);
 }

 //Load the test data from the file
 IDataView testData = mlContext.Data.LoadFromTextFile<TextData>(testDataPath, hasHeader: true, separatorChar: ',');

 //Apply the model to the test data
 var predictor = mlContext.Model.CreatePredictionEngine<TextData, SentimentPrediction>(model);

 //Get the predictions
 var testDataList = mlContext.Data.CreateEnumerable<TextData>(testData, reuseRowObject: false).ToList();
 foreach (var data in testDataList)
 {
     //Make a prediction
     var prediction = predictor.Predict(data);
     //Print the prediction
     Console.WriteLine($"Text: {data.text} | Prediction: {(prediction.IsPositiveSentiment ? "Positive" : "Negative")}");
 }

```

### Full code of the project 
```c#
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using Microsoft.ML.Trainers;
using System.Runtime.CompilerServices;
using System.ComponentModel;
using System.Reflection.Emit;

namespace Classification
{
    public class MovieReview
    {
        //The load column attribute is used in Ml.net to specify the index of the column in a data set to load into a property of a class when loading data from a file.
        //In this case, the attribute will load values from the first column index zero of the dataset file into the label property.

        [LoadColumn(0)]
        public string text { get; set; }

        [LoadColumn(1)]
        [ColumnName("Label")]
        public bool sentiment { get; set; }


    }

    public class TextData
    {
        [LoadColumn(0)]
        public string text { get; set; }
    }

    public class SentimentPrediction
    {
        [ColumnName("Score")]
        public float SentimentScore { get; set; }

        public bool IsPositiveSentiment => SentimentScore < 0.5f;
    }

        public class Program
    {
        public static void Main(string[] args)
        {
            ////Create a new MLContext instance
            //MLContext mLContext = new MLContext();
            ////Load the data from the file into an IDataView object
            //string dataPath = "train.csv";
            ////string text = File.ReadAllText(dataPath);
            ////using (StreamReader streamReader = new StreamReader(dataPath))
            ////{
            ////    text = text.Replace("\'", "");
            ////} 

            ////File.WriteAllText(dataPath, text);


            //IDataView dataView = mLContext.Data.LoadFromTextFile<MovieReview>(dataPath, hasHeader: true, allowQuoting: true, separatorChar: ',');

            ////Console.WriteLine("Data loaded successfully");
            ////Console.WriteLine();

            ////var preview = dataView.Preview();
            ////foreach (var row in preview.RowView)
            ////{
            ////    Console.WriteLine($"{row.Values[0]} | {row.Values[1]}");
            ////}

            ////The pipeline starts with the Transforms.Text.FeaturizeText method.
            ////This method is used to convert the text data into numerical features that can be used by the machine learning algorithm.
            ////It takes two parameters: the name of the output column ("Features") and the name of the input column ("text").
            ////The Append method is then called on the pipeline to add another component to the sequence.
            ////In this case, it appends the BinaryClassification.Trainers.SdcaLogisticRegression method, which represents the chosen machine learning algorithm.
            ////This algorithm is a binary logistic regression model trained using the Stochastic Dual Coordinate Ascent(SDCA) optimization algorithm.
            ////It takes two parameters: the name of the label column("Label") and the name of the feature column("Features").
            ////in summary, this line of code creates a pipeline that first converts the text data into numerical features and then applies a binary logistic regression model to train the data.
            //var pipeline = mLContext.Transforms.Text.FeaturizeText("Features", "text")
            //    .Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));

            //    //In machine learning, a pipeline is a sequence of data processing components, called transformers and estimators, that are applied in a specific order to transform the data and train a model.
            //    //The Fit method is used to train the model by fitting the pipeline to the data.
            //    //In this case, the pipeline variable represents the sequence of transformations and the chosen machine learning algorithm.
            //    //The Fit method takes the dataView as input and trains the model by applying the transformations and the chosen algorithm to the data.
            //var model = pipeline.Fit(dataView);

            ////The Transform method is used to apply the trained model to new data and generate predictions.
            //var predictions = model.Transform(dataView);

            ////The Evaluate method is used to evaluate the model's performance on the test data.
            //var metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");
            //Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            //Console.WriteLine($"Accuracy: {metrics.PositivePrecision}");
            //Console.WriteLine($"Accuracy: {metrics.PositiveRecall}");
            //Console.WriteLine($"Accuracy: {metrics.F1Score}");

            ////Save the model to a file
            //mLContext.Model.Save(model, dataView.Schema, "sentiment_model.zip");

            string modelPath = "sentiment_model.zip";
            string testDataPath = "movieReviewsTesting.csv";
            var mlContext = new MLContext();
            ITransformer model;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                //Load the model from the file
                model = mlContext.Model.Load(stream, out var modelInputSchema);
            }

            //Load the test data from the file
            IDataView testData = mlContext.Data.LoadFromTextFile<TextData>(testDataPath, hasHeader: true, separatorChar: ',');

            //Apply the model to the test data
            var predictor = mlContext.Model.CreatePredictionEngine<TextData, SentimentPrediction>(model);

            //Get the predictions
            var testDataList = mlContext.Data.CreateEnumerable<TextData>(testData, reuseRowObject: false).ToList();
            foreach (var data in testDataList)
            {
                //Make a prediction
                var prediction = predictor.Predict(data);
                //Print the prediction
                Console.WriteLine($"Text: {data.text} | Prediction: {(prediction.IsPositiveSentiment ? "Positive" : "Negative")}");
            }
        }
    }
}



```