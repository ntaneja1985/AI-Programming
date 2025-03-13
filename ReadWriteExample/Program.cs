using CsvHelper;
using System.Globalization;
using Newtonsoft.Json;
using System.IO;
using System.Collections.Generic;


class Program
{
    static void Main()
    {
        string csvFilePath = @"readAndWriteSampleCSV.csv";
        using (var reader = new StreamReader(csvFilePath))
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
        {
            var records = csv.GetRecords<MyData>();
            foreach (var record in records)
            {
                record.DisplayInfo();
            }
        }

        var dataToWrite = new List<MyData>() { new MyData { Name = "John", Age = 25, City = "New York" }
                                             ,new MyData { Name = "Jane", Age = 30, City = "Los Angeles" }
                                             ,new MyData{Name = "Bob", Age = 25, City = "Chicago"} };

        string outputCsvFilePath = @"output.csv";
        using (var writer = new StreamWriter(outputCsvFilePath))
        using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
        {
            csv.WriteRecords(dataToWrite);
        }

        string jsonFilePath = @"example.json";
        var jsonData = File.ReadAllText(jsonFilePath);
        var dataFromJson = JsonConvert.DeserializeObject<List<MyData>>(jsonData);
        foreach(var item in dataFromJson)
        {
            item.DisplayInfo();
        }

        var dataToWriteJson = new List<MyData>() { new MyData { Name = "John", Age = 25, City = "New York" }
                                             ,new MyData { Name = "Jane", Age = 30, City = "Los Angeles" }
                                             ,new MyData{Name = "Bob", Age = 25, City = "Chicago"} };

        string outputJsonFilePath = @"outputFile.json";
        var json = JsonConvert.SerializeObject(dataToWriteJson);
        File.WriteAllText(outputJsonFilePath, json);
    }
}

public class MyData
{
    public string Name { get; set; }
    public int Age { get; set; }
    public string City { get; set; }

    public void DisplayInfo()
    {
        Console.WriteLine($"Name: {Name}, Age: {Age}, City: {City}");
    }
}