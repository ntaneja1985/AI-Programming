using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Reflection;
using System.Text.RegularExpressions;
using static System.Runtime.InteropServices.JavaScript.JSType;

public class TextData
{
    public string Text { get; set; }
}

public class NgramPrediction
{
    public string Input { get; set; }
    public string Label { get; set; }
}

public class Program
{
    // This dictionary stores the n-gram model. The key is an n-gram, and the value is a dictionary that stores the frequency of each next word for the given n-gram key.
    // The ngramModel dictionary in the CreateNgramModel function serves as the core data structure for storing the n-gram model.
    // Its purpose is to keep track of the frequency of each possible next word given a specific sequence of n-1 words (the n-gram key).
    static Dictionary<string,Dictionary<string,int>> ngramModel = new Dictionary<string, Dictionary<string, int>>();

    static IEnumerable<TextData> LoadData(MLContext mlContext, string inputPath)
    {
        string text = File.ReadAllText(inputPath);
        return new List<TextData> { new TextData { Text = text } };
    }

    /*
     This function processes the text data, extracts n-grams, and builds a dictionary-based model that stores the frequency of each next word for a given n-gram key. 
     This model can be used to generate text or make predictions based on the provided data.
     */
    static void CreateNgramModel(IEnumerable<TextData> data, int n)
    {
        foreach (var text in data)
        {
            // Remove special characters and split the text into words
            // It removes any non-alphanumeric characters from the text using a regular expression.
            // It converts the text to lowercase.
            // It splits the text into an array of words using spaces, newlines, and carriage returns as delimiters.
            
            var words = Regex.Replace(text.Text, @"[^\w\s]", "").ToLower().Split(new char[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

            // Iterate over the array of words, excluding the last n words.
            for (int i = 0; i < words.Length - n; i++)
            {
                // Extract the n-gram key and the next word.
                var ngramKey = string.Join(" ", words.Skip(i).Take(n - 1));

                // Extract the next word.
                var nextWord = words[i + n - 1];

                // If ngram model doesnot contain the word, add it to the dictionary as the key
                if (!ngramModel.ContainsKey(ngramKey))
                {
                    ngramModel[ngramKey] = new Dictionary<string, int>();

                }

                //if the ngram model contains an entry with the key of the current word and its dictionary of next words contains the next word, increment it
                //we are doing all this to calculate the frequency of the next words and this will give us the most likely next word in the sequence.
                if (!ngramModel[ngramKey].ContainsKey(nextWord))
                {
                    ngramModel[ngramKey][nextWord] = 0;
                }
                ngramModel[ngramKey][nextWord]++;
            }
        }
    }

    // Randomly select a word from the dictionary based on the frequency of each word.
    // This method can be used in the context of text generation or prediction based on n-gram models.
    // By randomly selecting the next word from the dictionary of possible next words, it adds an element of randomness to the generated or predicted text.
    static string GetRandomWord(Dictionary<string, int> nextWords)
    {
        var total = nextWords.Values.Sum();
        var randomValue = new Random().Next(0, total);
        foreach (var word in nextWords)
        {
            randomValue -= word.Value;
            if (randomValue < 0)
            {
                return word.Key;
            }
        }

        return string.Empty;
    }

    // Generate text based on the provided initial text and length.
    static string GenerateText(string seed, int length)
    {
        var result = seed;
        var words = seed.Split(' ');
        for (int i = 0; i < length; i++) 
        { 
            var ngramKey = string.Join(" ", words.Skip(Math.Max(0,words.Length - 1)));
            // Check if the n-gram key is present in the model
            if (ngramModel.ContainsKey(ngramKey))
            {
                // Get the possible next words for the n-gram key
                var nextWords = ngramModel[ngramKey];
                // Get the next word based on the frequency of each word
                var nextWord = GetRandomWord(nextWords);
                // Add the next word to the result
                result += " " + nextWord;
                // Add the next word to the array of words
                Array.Resize(ref words, words.Length + 1);
                words[words.Length - 1] = nextWord;
            }
            else
            {
                break;
            }  
        }
        return result;
    }
        public static void Main(string[] args)
    {
        var mlContext = new MLContext();
        var data = LoadData(mlContext, "input.txt");

        // Create a 2-gram model
        /*
         Suppose n is 3, and the text is "the quick brown fox jumps over the lazy dog".
         The function will generate 3-grams like "the quick brown", "quick brown fox", "brown fox jumps", etc.
         For the 3-gram "the quick brown", "the quick" is the key, and "brown" is the next word.
         The ngramModel will be updated to reflect that "brown" follows "the quick" once.
         The ngramModel can be used to predict the next word in a sequence by looking up the n-gram key and selecting the most frequent next word.
         It can also be used to generate text by repeatedly predicting the next word based on the current sequence of words.
         The ngramModel dictionary is essential for storing the relationships between n-grams and their subsequent words, allowing the function to build a predictive model based on the input text data.
         */
        CreateNgramModel(data, n:2);
        Console.WriteLine("Enter a starting sentence:");
        string seed = Console.ReadLine();
        Console.WriteLine("Enter the length of the generated text:");
        int length = int.Parse(Console.ReadLine());
        string generatedText = GenerateText(seed, length);
        Console.WriteLine("\nGenerated Shakespearean text:");
        Console.WriteLine(generatedText);

    }
}