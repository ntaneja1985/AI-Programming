using MathNet.Numerics.Statistics;

public class House
{
    public int Bedrooms { get; set; }
    public int Bathrooms { get; set; }
    public double Size { get; set; }
    public double Price { get; set; }
    public double NormalizedSize { get; set; }

    public string Category { get; set; }
    public int CategoryEncoded { get; set; }

    // New Feature
    // Provide additional insights to our model
    public double SizePerBedroom { get; set; }
}

public class Program
{
    public static void Main()
    {
        #region CreateNewFeaturesFromExistingData
        List<House> houses = new List<House>() { new House
        {
            Bedrooms = 3,
            Bathrooms = 2,
            Size = 1500
        },
        new House
        {
            Bedrooms = 4,
            Bathrooms = 3,
            Size = 2000
        },
new House
        {
            Bedrooms = 2,
            Bathrooms = 1,
            Size = 1000
        }};

        foreach (House house in houses)
        {
            house.SizePerBedroom = house.Size / house.Bedrooms;
        }

        foreach (House house in houses)
        {
            Console.WriteLine("Bedrooms: " + house.Bedrooms + " Bathrooms: " + house.Bathrooms + " Size: " + house.Size + " SizePerBedroom: " + house.SizePerBedroom);
        }

        #endregion

        #region FeatureSelectionAndCorrelation
        double[] sizes = { 1500, 2000, 1000 };
        double[] prices = {300000, 400000, 200000 };
        // Calculate the correlation between size and price
        // A high correlation indicates that size and price are closely related
        //It also indicates the feature is relevant to the model
        double correlation = Correlation.Pearson(sizes, prices);
        Console.WriteLine("Correlation between size and price: " + correlation);
        #endregion

        #region FeatureTransformation
        List<House> housesForTransformation = new List<House>() { new House
        {
            Bedrooms = 3,
            Bathrooms = 2,
            Size = 1500,
            Price = 300000,
            Category = "Single Family"
        },
        new House
        {
            Bedrooms = 4,
            Bathrooms = 3,
            Size = 2000,
            Price = 400000,
            Category = "Condo"
        },
new House
        {
            Bedrooms = 2,
            Bathrooms = 1,
            Size = 1000,
            Price = 200000,
            Category = "Townhouse"
        }};

        double maxSize = housesForTransformation.Max(h => h.Size);
        double minSize = housesForTransformation.Min(h => h.Size);
        foreach(var house in housesForTransformation)
        {
            house.NormalizedSize = (house.Size - minSize) / (maxSize - minSize);
            Console.WriteLine("Size: " + house.Size + " Normalized Size: " + house.NormalizedSize);
        }


        var categoryEncoding = new Dictionary<string, int>
        {
            {"Single Family", 0},
            {"Condo", 1},
            {"Townhouse", 2}
        };

        foreach (var house in housesForTransformation)
        {
            house.CategoryEncoded = categoryEncoding[house.Category];
            Console.WriteLine("Category: " + house.Category + " Category Encoded: " + house.CategoryEncoded);
        }


        #endregion


    }
}

