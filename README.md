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