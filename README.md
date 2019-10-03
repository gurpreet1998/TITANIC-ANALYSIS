# TITANIC-ANALYSIS
### NOTE: This is over-view, refer to the notebook for deeper insights
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
 
 #### Method for calculating Age missing values :
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
### 1) SURVIVED vs DEAD
Now, lets start analyzing the data and get some useful insights from it.
* Lets see total number of people survied vs dead
  ![Survived/Dead](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_9_1.png)
   
   ##### NOTE : [ 0 : DEAD  1 :Survived ]
   
   ###### `'Passenger's survival Percentage`' = 38.38 %
   ###### `'Passenger's  Death Percentage`' = 61.62 %
   
### 2) 'Survivals ' vs 'P_class
This is to analyze if there is some trend in prediction survival related to class. There is chance that Higher class passenger were given more privilege to evacuate the ship.

* Below figure will show the Survived/Dead passengers according to Class

   !["CLASS vs SURVIVAL"](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_14_1.png)
   ##### From the above figure it is clearly visible that 'Class 1 ' passengers were able to survive followed by 'Class 2 'and 'Class 3 ' passengers. This trend is there may be because of the priority, higher class passengers will be allowed to evacuate first then followed by other classes.
   

* Lets, check the distribution of 'Males' and 'Females' in each class and there survival rate
  !["CLASS vs SURVIVAL"](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_16_1.png)
  
  ##### The above figure shows the similar pattern like that of P_class vs survival. This figure represents that females were given more importance then men in there respective class. Similiar is applicable to men.
  
## Question 1:
#### Is there chance that High class passangers were able to escape due to the embarkment location and not due to the status ? 

## Answer 1:
TO answer this question, Two things have to be done:
 1) First of all, analyze the graph of `'Embarked`' vs `'Fare`' 
    We are doing this because, we want to know which Embakement is having more Fare and what is the range of Fare.
    
      !["EMBARKED vs FARE"](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_19_1.png)
      !["EMBARKED vs FARE"](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_21_2.png)
      
#### Following are the analysis from the figures above:
###### LEFT FIGURE:
* 'Embarkemt C ' is mostly assigned to 'Class 1 ' passengers
* 'Embarkemt S ' is mostly assigned to 'Class 2 ' passengers
* 'Embarkemt Q ' is mostly assigned to 'Class 3 ' passengers
###### RIGHT FIGURE:
 This figure shows that Embarked C passengers were the ones who survived the most. This is not co-incident. Again hiher class were given more priority.
 
 
 ## Question 2:
 #### Is there relation between survival rate according to Fare and Age group?
  ## Answer 2:
  !["FARE vs Passengers Survived"](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_24_2.png)
  
  * From the above two parallel figure following observations are
made:
   ##### 'FARE ' VS 'SURVIVAL ' OBSERVATION:
  1) Passanger who paid more were survied more than passangers who paid less 
  
   ##### 'AGE ' vs 'SURVIVAL ' OBSERVATION
   1) Talking about age, almost all old age passangers Age[70-80] died.
   2) Second most deaths were recorded in age group [20-40].
   
   #### Deep analysis of deaths according to age groups
   Lets divide the age into 4 standard age groups, then we will look that in which group Death is recorded more.
   ###### Distribute the age groups into 4 category :
     *  `'Child`' ------------(0-12) Years
     *  `'Adolescence`'-------(13-18) Years
     *  `'Adult`'-------------(19-49) Years
     *  `'Senior_Adult`' -----(50 years and above)
     
   !["Age group vs Survived"](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_39_2.png)
  
  #### Following are the observation from above plot
  * Among the Age groups, `'childrens`' were given more priority.
  * `'Adolescent`' were rescued after childrens
  * Most deaths are in `'Adult`' and `'Senior_Adult`' group
  
  
     
 # Adding New Features:
 ### Feature 1:
 * There is a term called 'Synergy effect ' or 'Interaction ' between features. This will lead to
justify the increase in the value of one feature due to the per unit change in another feature. For example,

                           Y = β0 + β1 ∗ X1 + β2 ∗ X2 + error.
                        
* According to this model, if we increase X 1 by one unit, then Y will increase by an average of β 1 units.
Notice that the presence of X 2 does not alter this statement—that is, regardless of the value of X 2 , a
one-unit increase in X 1 will lead to a β 1 -unit increase in Y . One way of extending this model to allow for
interaction effects is to include a third predictor, called an interaction term, which is constructed by
computing the product of X 1 and X 2 . This results in the model

                        Y = β0 + β1 ∗ X1 + β2 ∗ X2 + (β3 ∗ X1 ∗ X2) + error
                        
     
    
  ### Feature 2:
  * Now we know that 'Sibsp ' resembles Siblings and 'Parch ' represents parents. So accordingly we can
find total family members. We are adding this feature because there is chance that someone having family get died to save them.
 
  
                         df['Family_Size']=df['SibSp']+df['Parch']        
                         
 ### Feature 3:
 *'Age_group ' feature is added during analysis of the age vs survival. $ categorical feature collumn is added that we have discussed earlier.
 
 
 # Normalizing The Data
 As quantitative values are having a little difference. For exmaple - ' FARE' , ' AGE' have large values as
compared to rest of features. This condition will make algorithm to put more weights to those features.
Therefore, to deal with this situation NORMALIZATION is must. For this we ususally use - `'MIN_MAX_SCALER`'

  !["Age group vs Survived"](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_86_0.png)
  
  !["Age group vs Survived"](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_90_0.png)
                    
### ------------------------------------------------------------------------------------------------------------------------
## Model Formulation

### Till now we have get intuition from the data, Now lets use predictive models to predict the survival.
 In this notebook we have used 3 Supervised model for classification
 
 * K-Nearest Neighbors (KNeighbors)
 * Support Vector Machines (SVM)
 * Quadractic Discriminant Analysis
 
 !["Age group vs Survived"](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/output_106_4.png)
   !["Age group vs Survived"](https://github.com/gurpreet1998/TITANIC-ANALYSIS/blob/master/Output%20figures/metrics%20table)
 
### Analysis Of Result

* As we see in the output histograms,and Table 'KNN ' does the best among the three.

* There is huge difference between model's 'trainng accuracy' and ' Testing_accuracy' . This shows that model is over fitting on the data.

* To deal with the over fitting problem we can remove certain features that are irrelevent to the model.
      
* To deal with Overfitting we can do Regularization.
   

## KAGGLE SCORE: 0.7755 
 

 

 
