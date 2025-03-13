using ScottPlot;
using ScottPlot.Palettes;

public class Sale
{
    public string Category { get; set; }
    public double Amount { get; set; }
}
class Program
{
    static void Main()
    {
        //Plot plot = new Plot();
        //double[] dataX = { 1, 2, 3, 4, 5};
        //double[] dataY = { 2, 4, 6, 8, 10 };

        //plot.Title("Title of the graph");
        //plot.XLabel("X values");
        //plot.YLabel("Y values");
        //plot.Add.Scatter(dataX,dataY);
        //plot.SavePng("demo.png",500,500);

        List<Sale> sales = new List<Sale>()
        {
            new Sale() { Category = "Electronics", Amount = 1000 },
            new Sale() { Category = "Fashion", Amount = 1500 },
            new Sale() { Category = "Cosmetics", Amount = 500 },
            new Sale() { Category = "Electronics", Amount = 2000 },
            new Sale() { Category = "Fashion", Amount = 2500 },
            new Sale() { Category = "Cosmetics", Amount = 1000 },
            new Sale() { Category = "Electronics", Amount = 3000 },
            new Sale() { Category = "Fashion", Amount = 3500 },
            new Sale() { Category = "Cosmetics", Amount = 2000 },
        };

        var summary = sales.GroupBy(s => s.Category)
            .Select(g => new
            {
                Category = g.Key,
                TotalSales = g.Sum(s => s.Amount),
                Average = g.Average(s => s.Amount),
                Min = g.Min(s => s.Amount),
                Max = g.Max(s => s.Amount)
            }).ToList();

        foreach (var item in summary)
        {
            Console.WriteLine($"Category: {item.Category}, Total Sales: {item.TotalSales}");
        }
    }
}