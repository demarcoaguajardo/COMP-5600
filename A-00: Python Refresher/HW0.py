# Importing Libraries
import numpy
import matplotlib.pyplot as plot

# ---------- Problem 1 ----------

print("---------- Problem 1 ----------\n")

# Step 1: Create 1D NumPy array of shape [1x5] with random values from a
# uniform distribution.

# Create 1D array of size 5 holding values from 0 to 1.
array1D = numpy.random.uniform(0, 1, 5)
# Prints 1D array's contents.
print("1D NumPy Array: ", array1D)
print("\n")

#Step 2: Compute mean and standard deviation of the array.

# Get mean of array.
mean = numpy.mean(array1D)
# Get standard deviation of array.
stdeviation = numpy.std(array1D)
# Prints mean and standard deviation of array.
print("Mean: ", mean)
print("Standard Deviation: ", stdeviation)
print("\n")

# Step 3: Reshape array into a 2D array with 5 rows and 1 column.

# Reshape 1D array into 2D array with 5 rows and 1 column.
array2D = array1D.reshape(5, 1)
# Prints 2D array's contents.
print("2D NumPy Array:\n", array2D)
print("\n")

# Step 4: Add 5 to each element in the array and print the resulting array.

# Add 5 to each element in the array.
array2D += 5
# Print new result.
print("2D NumPy Array after adding 5 to each element:\n", array2D)
print("\n")


# Step 5: Compute Dot Product of the new 2D array 
dotProduct = numpy.dot(array2D.T, array2D)
# Print result of dot product.
print("Dot Product of 2D Array:\n", dotProduct)

# ---------- Problem 2 ----------

# Step 1: Generate set of x values from 0 to 100 with an increment of 0.1.
xValues = numpy.arange(0, 100, 0.1)

# Step 2: Compute corresponding y values using function y = sin(x). 
yValues = numpy.sin(xValues)

# Step 3: Plot sine wave using Matplotlib, adding labels for x and y axes,
# as well as a title for the plot.

# Plots the sine points.
plot.plot(xValues, yValues)
# Adds labels and title.
plot.xlabel("X Values")
plot.ylabel("Y Values")
plot.title("Sine Graph")


# Step 4: Save plot as a .png file named "sine_wave.png".
plot.savefig("sine_wave.png")

# ---------- Problem 3 ----------

# Step 1: Create two NumPy arrays; one for x values from 0 to 100 (inc 1).
# two for y values using y = 0.5x^(2) + 2x + 1.
newXValues = numpy.arange(0, 100, 1)
newYValues = 0.5 * (newXValues ** 2) + 2 * newXValues + 1

# Step 2: Plot quadratic function using Matplotlib, adding labels and legend.
# Includes my chosen linestyle.
plot.plot(newXValues, newYValues, label='y = 0.5x^2 + 2x + 1', linestyle = "-.")
plot.xlabel("X Values")
plot.ylabel("Y Values")
plot.title("Quadratic Function")
plot.legend()

# Step 3: Add gridlines to plot.
plot.grid(True)

# Step 4: Save plot as a pdf file named "quadratic_function.pdf".
plot.savefig("quadratic_function.pdf")
