---
layout: default
title: Exploración de Datos
permalink: /exploracion/
---
# Data Exploring
First of all, we must understand the nature of the data set, we need to know things like, how many data are avaliable in the dataset, what kind of variables we are dealing with, we must know if they are numbers, words or any other kind of information.

### Disclaimer
As mentioned before, we are going to use python programming + AI tools, most specifically in google colab due to the flexibility and easygoing use, this is just to mention the tools we're going to use for this study, but it's not necessary to deeply know about it cause this is a general study wich only contains procedure and results, **no coding included**.

### Data Loading
Lets load the data set and see how many information we have
![Data Loading](/assets/module1/resources/Data_loading.png)

The data set appears to have 2111 rows and 10 colums size, wich is a good size for a dataset, now let's see what kind of variables we're going to be working with
### Variables and types of information
![Data type](/assets/module1/resources/Data_type.png)

As we appreciate, we have two types of variables, "float64" and "Object", variables which for technical purposes we're going to express as quantitative (numbers) and qualitative (categories) respectively.

# General Aspects
Just by looking at it, the dataset seems quite general and basic, nothing very specific; after all, we only have ten variables to work with. Furthermore, some variables seem to be subjective rather than specific, variables like *"ComeMuchasCalorias"*. Considering the human factor, we would need to establish a range of how many calories are considered a lot and how many are considered less. Regardless, we must also consider the fact that the average person in Colombia, Mexico, and Peru is not able to answer a specific range of calories; therefore, it seems appropriate to treat this variable as a category rather than a number. However, just suggesting, it would be more specific to use categories like *(High/Normal/Insufficient).*

Other variables also seem to be very subjective, such as *"ComeVegetales"* and *"ConsumoDeAgua"*, because the dataset does not specify if the numbers contained in both variables have a measure unit, therefore seems correct to treat them as a relative consume, but it would be really helpful if it specifies the numbers as weekly frequency, liters, gallons or any other unit.

Now, if we talk about obesity levels, we can put our attention to that variable which seems to be categoric, just to see how many people with different levels of obesity, we're going to plot the data to see the distribution. 

![Obesity distribution](/assets/module1/resources/Obesity_levels_categories.png)

According to the graph, obesity levels appear to be fairly well distributed across its different categories, with type 1 obesity being the most common between the data.

Now, instinctively, there must be a tendency of obesity according to genre, let's explore the distribution based on genre to see if males or females tend to be in a specific category of obesity, let's see if the previous are on the right way.

![Obesity genre](/assets/module1/resources/Obesity_type_genre.png)

Looking closely at the plot, we can see a striking pattern in severe obesity levels: Type II is dominated by males, while Type III is almost exclusively female. In contrast, the other weight categories show a much more balanced distribution between sexes.
