# TITANIC-ANALYSIS
This repository aim to provide competitive analysis of the data that help us to know the facts and eextract the meaning full information that lead to predict the survival of the passengers.
In this contest, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

### Libraries:
* Pandas
* Numpy
* Scikit-Learn
* Matplotlib

### Nokebook Aim:
#### Importing specific libraries
* Listed above

#### Data cleaning:
Titanic data consist of following features:
1) `'PassengerId`'  --- text
2) `'Survived`'     --- categorical
3) `'Pclass`'       --- Categ
4) `'Name`'         --- text
5) `'Sex`'          --- Categorical
6) `'Age`'          --- Continuous 
7) `'SibSp`'        --- continuous
8) `'Parch`'        --- continuous
9) `'Ticket`'       --- Text
10) `'Fare`'        --- continuous
11) `'Cabin`'       --- text
12) `'Embarked`'    --- categorical
##### IN TRAINING DATA
* `'Age`'   :- This attribute has `'19.87%`' null values
* `'Cabin`' :- This attribute has most of the missing values `'77.10 %`'
##### IN TEST DATA 
* `'Cabin`' : `'78.23 %`' missing values
* `'Age`'   : `'20.57 %`' missing values
* `'Fare`'  : `'0.24 %`'

As per the given data, Cabin has most of its values missing, so it is wise to drop this feature.
Now for `'Age`', it is looking as important feature for further computation and analysis, so there are three ways to deal with this :
 1) USE `'MEAN`' , `' MODE`' from whole coulmn
 2) Analyse the age with other columns like `'Pclass`' and `'Sex`' , check the distribution of the ages in
Pclass and Sex in Pclass. Accorindly we can add the values.
 3) We can make a seperate regression model as predict the values of the age.
 
 As of now, we has choose 2nd method
 
 #### Method for calculating mean Age missing values :
 In this approach, Passsengers are grouped into Sex according to there respective class then Mean Age is recorded for each group of people.
 See this figure below to have better understanding. 
![AGE-CLASS-SEX](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_51_1.png)

### For calulating Fare missing value:
Beacause it is only one missing value , we noticed that value to which class it belong to (Pclass = 3), accordingly we calulated the mean of the Fare of class 3. Then assigned the value to missing spot.
   * Mean_value is = $ 12.46
   which is quit justifiable.

## Label Encoding
As we know that some of features have categorical values, so it is better to convert it using Label_encoder. 

# DATA ANALYSIS
### SURVIVED vs DEAD
Now, lets start analyzing the data and get some useful insights from it.
1) Lets see total number of people survied vs dead
   ![Survived/Dead](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_9_1.png)
   ##### NOTE : [ 0 : DEAD  1 :Survived ]
   
   ###### `'Passenger's survival Percentage`' = 38.38 %
   ###### `'Passenger's  Death Percentage`' = 61.62 %



   




 

 

 
