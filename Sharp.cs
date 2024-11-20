using System;

namespace DemoApplication
{
    // This is a simple class to demonstrate basic C# functionality.
    public class Program
    {
        // The entry point of the C# application.
        public static void Main(string[] args)
        {
            Console.WriteLine("Welcome to the C# Demo!");

            // Example of a method call that returns a value.
            int result = AddNumbers(5, 10);
            Console.WriteLine("The result of adding 5 and 10 is: " + result);

            // Example of a method with user input.
            Console.Write("Enter your name: ");
            string name = Console.ReadLine();
            GreetUser(name);

            // Example of a loop (printing numbers).
            Console.WriteLine("\nCounting from 1 to 5:");
            for (int i = 1; i <= 5; i++)
            {
                Console.WriteLine(i);
            }

            // Example of a conditional statement.
            Console.Write("\nEnter your age: ");
            int age = int.Parse(Console.ReadLine());
            CheckVotingEligibility(age);

            Console.WriteLine("\nPress any key to exit.");
            Console.ReadKey();
        }

        // Method to add two integers and return the result.
        public static int AddNumbers(int a, int b)
        {
            return a + b;
        }

        // Method to greet the user.
        public static void GreetUser(string userName)
        {
            Console.WriteLine($"Hello, {userName}!");
        }

        // Method to check if a person is eligible to vote.
        public static void CheckVotingEligibility(int age)
        {
            if (age >= 18)
            {
                Console.WriteLine("You are eligible to vote.");
            }
            else
            {
                Console.WriteLine("You are not eligible to vote yet.");
            }
        }
    }
}
