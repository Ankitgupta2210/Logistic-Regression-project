# Logistic-Regression-project
# Logistic-Regression-Social-Network-Ads
Using Logistic Regression model to predict if a person is going to buy a new car or not based on the available data

# Problem
In a car company, a survey was done through social media advertising, in which companies want to find out which person is interested in buying a car and which is not?

# Dataset
The dataset contains 400 entries which contains the userId, gender, age, estimatedsalary and the purchased history. The matrix of features taken into account are age and estimated salary which are going to predict if the user is going to buy new car or not(1=Yes, 0=No).

# Solution
First to import libaray and upload the dataset and performe the some opertion .

step 1:- Date preproocessing.

step 2:- Apply the logistic regression

step 3 :- Evaluate the prediction

step 4:- visualization 

The confusion matrix below shows that our model predicts 90 correct and 10 wrong decisions which shows 89% accuracy

 
array([[66,  2],
       [ 8, 24]], dtype=int64)
       
       
       
The data visualization of the training set and test set is given below. As logistic regression is linear model the data is being separated linearly. The blue dots shows the people buying the car whereas yello dots shows the people who don't buy the car.

![image](https://user-images.githubusercontent.com/102615860/201536553-b6f987d5-40a1-4147-a010-e04319aa15b4.png)
       


