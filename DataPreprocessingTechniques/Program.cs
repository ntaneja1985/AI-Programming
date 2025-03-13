using System;
using System.Collections.Generic;
using System.Linq;


public class MyData
{
    public string Name { get; set; }
    public int? Age { get; set; }
}

public class ScaledData
{
    public string Name { get; set; }
    public double Age { get;set; }
}

public class MyDataCategorical
{
    public string Name { get; set; }
    public string Category { get; set; }
}

    public class Program
{
    public static void Main()
    {
        List<MyData> list = new List<MyData>();
        list.Add(new MyData { Name = "John", Age = 25 });
        list.Add(new MyData { Name = "Diana", Age = null });
        list.Add(new MyData { Name = "Jane", Age = 30 });
        list.Add(new MyData { Name = "Bob", Age = null });

        List<MyData> cleanedData  = list.Where(x => x.Age.HasValue).ToList();
        double meanAge = list.Where(x => x.Age.HasValue).Average(x => x.Age.Value);
        List<MyData> imputedData = list.Select(x => new MyData { Name = x.Name, Age = x.Age ?? (int)meanAge }).ToList();
        Console.WriteLine("Data after removing missing values");
        cleanedData.ForEach(x => Console.WriteLine($"Name: {x.Name}, Age: {x.Age}"));
        Console.WriteLine("\nData after imputing missing values");
        imputedData.ForEach(x => Console.WriteLine($"Name: {x.Name}, Age: {x.Age}"));


        List<MyData> data = new List<MyData>();
        data.Add(new MyData { Name = "John", Age = 25 });
        data.Add(new MyData { Name = "Diana", Age = 30 });
        data.Add(new MyData { Name = "Jane", Age = 35 });
        data.Add(new MyData { Name = "Bob", Age = 40 });

        int minAge = data.Min(x => x.Age.Value);
        int maxAge = data.Max(x => x.Age.Value);

        List<ScaledData> scaledData = data.Select(x => new ScaledData { Name = x.Name, Age = (double)(x.Age.Value - minAge) / (maxAge - minAge) }).ToList();

        Console.WriteLine("\nData after scaling");
        scaledData.ForEach(x => Console.WriteLine($"Name: {x.Name}, Age: {x.Age}"));

        List<MyDataCategorical> dataCategorical = new List<MyDataCategorical>();
        dataCategorical.Add(new MyDataCategorical { Name = "John", Category = "A" });
        dataCategorical.Add(new MyDataCategorical { Name = "Diana", Category = "B" });
        dataCategorical.Add(new MyDataCategorical { Name = "Jane", Category = "A" });
        dataCategorical.Add(new MyDataCategorical { Name = "Bob", Category = "C" });

        var categories = dataCategorical.Select(x => x.Category).Distinct().ToList();
        var oneHotEncodedData = dataCategorical.Select(x => new
        {
            Name = x.Name,
            CategoryA = x.Category == "A" ? 1 : 0,
            CategoryB = x.Category == "B" ? 1 : 0,
            CategoryC = x.Category == "C" ? 1 : 0
        }).ToList();

        Console.WriteLine("\nData after one-hot encoding");
        oneHotEncodedData.ForEach(x => Console.WriteLine($"Name: {x.Name}, CategoryA: {x.CategoryA}, CategoryB: {x.CategoryB}, CategoryC: {x.CategoryC}"));
    }
}