public class Employee
{
    public int EmployeeId { get; set; }
    public string Name { get; set; }
}

public class Department
{
    public int EmployeeId { get; set; }
    public string DepartmentName { get; set; }
}

public class SalesRecord
{
    public string Product { get; set; }
    public double Price { get; set; }

    public int Quantity { get; set; }
}

    public class Program
{
    public static void Main()
    {
        List<Employee> employees = new List<Employee>()
    {
        new Employee { EmployeeId = 1, Name = "John Doe" },
        new Employee { EmployeeId = 2, Name = "Jane Doe" },
        new Employee { EmployeeId = 3, Name = "Sam Doe" }
    };

        List<Department> departments = new List<Department>()
    {
        new Department { EmployeeId = 1, DepartmentName = "HR" },
        new Department { EmployeeId = 2, DepartmentName = "IT" },
        new Department { EmployeeId = 3, DepartmentName = "Finance" }
    };

        var combinedData = from employee in employees
                           join department in departments
                           on employee.EmployeeId equals department.EmployeeId
                           select new
                           {
                               employee.EmployeeId,
                               employee.Name,
                               department.DepartmentName
                           };

        foreach (var data in combinedData)
        {
            Console.WriteLine($"EmployeeId: {data.EmployeeId},Name:{data.Name}, Department: {data.DepartmentName}  ");
        }

        List<SalesRecord> salesRecords = new List<SalesRecord>()
        {
            new SalesRecord { Product = "Laptop", Price = 1000, Quantity = 2 },
            new SalesRecord { Product = "Mobile", Price = 500, Quantity = 5 },
            new SalesRecord { Product = "Tablet", Price = 300, Quantity = 3 },
            new SalesRecord { Product = "Desktop", Price = 1500, Quantity = 1 }
        };

        //Group and apply aggregation functions
        //Group sales records by product and calculate total revenue and quantity
        var totalSales = salesRecords.GroupBy(s => s.Product).Select(s => new
        {
            Product = s.Key,
            TotalRevenue = s.Sum(p => p.Price * p.Quantity),
            TotalQuantity = s.Sum(p => p.Quantity)
        });

        foreach (var sales in totalSales)
        {
            Console.WriteLine($"Product: {sales.Product}, Total Revenue: {sales.TotalRevenue}, Total Quantity: {sales.TotalQuantity}");
        }
    }
}
