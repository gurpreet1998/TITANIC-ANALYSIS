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
1) `'PassengerId`'
2) `'Survived`'
3) `'Pclass`'
4) `'Name`'
5) `'Sex`'
6) `'Age`'
7) `'SibSp`'
8) `'Parch`'
9) `'Ticket`'
10) `'Fare`'
11) `'Cabin`'
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
 In this approach, Passsengers are into Sex according to there class and Age is recorded for each group of people (Refere to figure in Notebook) 
![Image description](link-to-image)


 

 

 
