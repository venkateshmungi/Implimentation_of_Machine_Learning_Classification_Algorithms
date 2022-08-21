#!/usr/bin/env python
# coding: utf-8

# <h3 style = "color:green"><center><i>IMPLIMENTATION OF ML-CLASSIFICATION MODELS<i></center></h3>

# <h3 style = "color:BlueViolet"><left><i>Importing the required Libraries and Loading the Dataset<i></left></h3>

# In[1]:


# Importing required libraries

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics


# In[2]:


# Loading the dat set

data = pd.read_csv(r"C:\\PYTHON\\PANDAS\\train.csv")


# In[3]:


# Displaying the Dataset
data


# <h3 style = "color:BlueViolet"><left><i>Copying the Dataset<i></left></h3>

# In[4]:


# Copying the dataset

df1 = data.copy()


# In[5]:


df1.head(3)


# <h3 style = "color:BlueViolet"><left><i>Deleting the Unwanted columns from the Dataset<i></left></h3>

# In[6]:


# Deleting the Unwanted columns from the data set

del df1["Loan_ID"]


# <h3 style = "color:BlueViolet"><left><i>Removing Special Charectors from the Feature "Dependents"<i></left></h3>

# In[7]:


# Seperating the unwanted symbols from the feature "Dependents"
import warnings 
warnings.filterwarnings("ignore")

df1['Dependents'] = df1['Dependents'].str.replace("+","")


# In[8]:


df1["Dependents"]


# - By doing the above step we removed "+" signs from the column: ["Dependents"]

# In[9]:


# Getting the shape of the data

df1.shape


# - We have 12 columns and 614 Rows in our dataset after deleting the column "Loan_ID"

# In[10]:


# Finding the datatypes of the data set

df1.dtypes


# - We have Seven categorical columns and Four Continuous types and One Discrete type of columns

# In[11]:


# Getting Metadata information of the given Dataset
df1.info()


# <h3 style = "color:BlueViolet"><left><i>Finding the "Statistics" of the Continuous Features<i></left></h3>

# In[12]:


df1.describe()


# - From the above information it is clear that The max income of applicant is 81000 and minimum income is 150 rupees
# - The Maximum Coapplicant-income is 41667 and minimum is 0
# - The Maximum Loan Amount is 700 and minimum is 9 rupees
# 
#   - There is Maximum chances to get loan , applicant who has high income or applicant who has his coapplicant's income high

# In[13]:


df1.mean()                    # Mean from Continuous columns


# In[14]:


df1.median()                 # Median from the Continuous columns


# In[15]:


df1.mode().sum()             # Mode from each column of the Dataset


# In[16]:


df1.std()                   # Standard deviation from the Continuous columns


# In[17]:


df1.var()                   # Variance of the Continuous columns


# In[18]:


from scipy.stats import skew         # importing skew module to get the skewness of the Continuous Data
from scipy.stats import kurtosis     # importing kutosis module to get the kurtosis of the Continuous Data


# In[19]:


df1.skew()


# In[20]:


df1.kurtosis()


# <h3 style = "color:BlueViolet"><left><i>Finding Minimum, Maximum, Range From the Dataset<i></left></h3>

# In[21]:


df1.min()


# In[22]:


df1.max()


# - We can't get range from String objects. So, now we will seperate Continuou feature to get range of our data

# In[23]:


# Seperating out the Continuous Features to get Range of our data
df2 = df1[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]]


# In[24]:


_range = df2.max() - df2.min()
_range


# <h3 style = "color:BlueViolet"><left><i>Value_Count on Categorical Features<i></left></h3>

# In[25]:


df1.Gender.value_counts()


# In[26]:


df1.Married.value_counts()


# In[27]:


df1.Dependents.value_counts()


# In[28]:


df1.Education.value_counts()


# In[29]:


df1.Self_Employed.value_counts()


# In[30]:


df1.Property_Area.value_counts()


# In[31]:


df1.Loan_Status.value_counts()


# - From the above information it is clear that females are 1/3 rd of males. It seems to be imbalanced data

# <h3 style = "color:blueviolet"><left><i>Checking for Missing Values<i></left></h3>

# In[32]:


df1.isnull().sum()              # Getting Missing values count from each column


# <h3 style = "color:BlueViolet"><left><i>Filling Missing Categorical Values with mode()<i></left></h3>

# In[33]:


df1['Gender'].fillna(df1.Gender.mode()[0], inplace = True)


# In[34]:


df1["Married"].fillna(df1.Married.mode()[0], inplace = True )


# In[35]:


df1["Dependents"].fillna(df1.Dependents.mode()[0], inplace = True )


# In[36]:


df1["Self_Employed"].fillna(df1.Self_Employed.mode()[0], inplace = True )


# In[37]:


df1["Credit_History"].fillna(df1.Credit_History.mode()[0], inplace = True )


# In[38]:


df1["LoanAmount"].fillna(df1.LoanAmount.mode()[0], inplace = True )


# In[39]:


df1["Loan_Amount_Term"].fillna(df1.Loan_Amount_Term.mode()[0], inplace = True )


# In[40]:


df1.isnull().sum()              # Re_Checking Missing values count from each column After Filled them


# <h3 style = "color:BlueViolet"><left><i>Applying Label Encoding On Categorical Features<i></left></h3>

# In[41]:


from sklearn.preprocessing import LabelEncoder


# In[42]:


encoder = LabelEncoder()


# In[43]:


df1["Gender"] = encoder.fit_transform(df1["Gender"])   # Encoding "Gender" Column


# In[44]:


print(df1["Gender"].unique())                          # Getting unique values of "Gender"


# In[45]:


df1["Married"] = encoder.fit_transform(df1["Married"]) # Encoding "Married" Column


# In[46]:


print(df1["Married"].unique())                         # Getting unique values of "Married"


# In[47]:


df1["Education"] = encoder.fit_transform(df1["Education"]) # Encoding "Education" column


# In[48]:


print(df1["Education"].unique())                           # Getting unique values of "Education"


# In[49]:


df1["Self_Employed"] = encoder.fit_transform(df1["Self_Employed"]) # Encoding "Self_Employed"


# In[50]:


print(df1["Self_Employed"].unique())                               # Getting unique values of "Self_Employed"


# In[51]:


df1["Loan_Amount_Term"] = encoder.fit_transform(df1["Loan_Amount_Term"]) # Encoding "Loan_Amount_Term"


# In[52]:


print(df1["Loan_Amount_Term"].unique())                                  # Getting unique values of "Loan_Amount_Term"


# In[53]:


df1["Property_Area"] = encoder.fit_transform(df1["Property_Area"]) # Encoding "Property_Area"


# In[54]:


print(df1["Property_Area"].unique())                               # Getting unique values of "Property_Area"


# In[55]:


df1["Loan_Status"] = encoder.fit_transform(df1["Loan_Status"]) # Encoding "Loan_Status"


# In[56]:


print(df1["Loan_Status"].unique())                             # Getting unique values of "Loan_Status"


# In[57]:


df1.head(3)


# <h3 style = "color:BlueViolet"><left><i>Checking for Outliers in Continuous Data<i></left></h3>

# In[58]:


plt.figure(figsize = (18,9))

plt.suptitle("OUTLIERS CHECKING BEFORE CAPPING",fontsize = 'x-large',weight = 'extra bold',ha = "center")
plt.subplot(2,3,1)
sns.boxplot(df1["ApplicantIncome"],color = "green")
plt.subplot(2,3,4)
sns.distplot(df1["ApplicantIncome"],color = "red")
plt.subplot(2,3,2)
sns.boxplot(df1["CoapplicantIncome"], color = "red")
plt.subplot(2,3,5)
sns.distplot(df1["CoapplicantIncome"], color = "blue")
plt.subplot(2,3,3)
sns.boxplot(df1["LoanAmount"],color = "blue")
plt.subplot(2,3,6)
sns.distplot(df1["LoanAmount"], color = 'green')
plt.show()


# <h3 style = "color:BlueViolet"><left><i>Outliers Capping Using Inter quartile Range Method<i></left></h3>

# In[59]:


Q1 = df1["ApplicantIncome"].quantile(0.25)

Q3 = df1["ApplicantIncome"].quantile(0.75)

IQR = Q3 - Q1

print(IQR)


# In[60]:


upperlimit = Q3 + 1.5 * IQR


# In[61]:


lowerlimit = Q1 - 1.5 * IQR


# In[62]:


df1["ApplicantIncome"] = np.where(df1["ApplicantIncome"] > upperlimit, upperlimit,
                         np.where(df1["ApplicantIncome"] < lowerlimit, lowerlimit,
                                  df1["ApplicantIncome"]))


# In[63]:


Q11 = df1["CoapplicantIncome"].quantile(0.25)

Q33 = df1["CoapplicantIncome"].quantile(0.75)

IQR1 = Q33 - Q11

print(IQR1)


# In[64]:


upperlimit1 = Q33 + 1.5 * IQR1


# In[65]:


lowerlimit1 = Q11 - 1.5 * IQR1


# In[66]:


df1["CoapplicantIncome"] = np.where(df1["CoapplicantIncome"] > upperlimit1, upperlimit1,
                           np.where(df1["CoapplicantIncome"] < lowerlimit1, lowerlimit1,
                                    df1["CoapplicantIncome"]))


# In[67]:


Q111 = df1["LoanAmount"].quantile(0.25)

Q333 = df1["LoanAmount"].quantile(0.75)

IQR2 = Q333 - Q111

print(IQR2)


# In[68]:


upperlimit2 = Q333 + 1.5 * IQR2


# In[69]:


lowerlimit2 = Q111 - 1.5 * IQR2


# In[70]:


df1["LoanAmount"] = np.where(df1["LoanAmount"] > upperlimit2, upperlimit2,
                    np.where(df1["LoanAmount"] < lowerlimit2, lowerlimit2,
                             df1["LoanAmount"]))


# In[71]:


plt.figure(figsize = (18,9))

plt.suptitle("OUTLIERS CHECKING AFTER CAPPING",fontsize = 'x-large',weight = 'extra bold',ha = "center")
plt.subplot(2,3,1)
sns.boxplot(df1["ApplicantIncome"],color = "green")
plt.subplot(2,3,4)
sns.distplot(df1["ApplicantIncome"],color = "red")
plt.subplot(2,3,2)
sns.boxplot(df1["CoapplicantIncome"], color = "red")
plt.subplot(2,3,5)
sns.distplot(df1["CoapplicantIncome"], color = "blue")
plt.subplot(2,3,3)
sns.boxplot(df1["LoanAmount"],color = "blue")
plt.subplot(2,3,6)
sns.distplot(df1["LoanAmount"], color = 'green')
plt.show()


# <h3 style = "color:BlueViolet"><left><i>Dealing with outliers in categorical data<i></left></h3>

# In[72]:


plt.figure(figsize = (20,4))
plt.suptitle("Difference Between Gender")
plt.subplot(1,4,1)
ax = df1["Gender"].plot.hist()
ax.set_ylabel("Frequecy")
ax.set_xlabel("Gender")
plt.subplot(1,4,2)
sns.barplot(x = "Gender", y = "Married", data = df1)
plt.subplot(1,4,3)
sns.barplot(x = "Gender", y = "Married", data = df1, hue = "Dependents", )
plt.subplot(1,4,4)
sns.barplot(x = "Married", y = "Gender", data = df1)
plt.show()


# - Here we can see that the count of "Female" is very less than "Male" and can be considerd as outlier

# In[73]:


plt.figure(figsize = (18,3))
plt.suptitle("OUTLIERS")
plt.subplot(1,5,1)
sns.boxplot(df1["Education"])
plt.subplot(1,5,2)
sns.boxplot(df1["Self_Employed"])
plt.subplot(1,5,3)
sns.boxplot(df1["Credit_History"])
plt.subplot(1,5,4)
sns.boxplot(df1["Loan_Amount_Term"])
plt.subplot(1,5,5)
sns.boxplot(df1["Property_Area"])
plt.show()


# - From the above plots it is conformed that categorical data also hsa outliers. The reasons for being outliers in the categorical data such as fault collection of the data or categories can be rare and hard to collect data about it. 
# 
#   The following are the ways to dealing with outliers in categorical data:
#          
#          1) Retention : This process involves modelling the outliers with the other data.
#          
#          2) Exclusion : This method involves techniques to exclude the outliers from the data.
#          
#          3) Replacement : Sometimes it happens that the data that is collected has outlier values but as a category, they are similar to the other major categories. In such cases, we can replace the outliers with similar categories. We can measure the similarity between the data using the measures like euclidean distance, cosine similarity, Manhattan distance etc.
#          
#          4) Sampling : Outliers in the categorical data can also be said to the problem of class imbalance. This means that the data for every class are not in a similar proportion. In such a situation, we use some of the sampling techniques such as downsampling, oversampling and SMOTE analysis. Here we mainly increase or decrease the data points by knowing the importance of the categories in the modelling. 
#          
# - Here I have choosen DownSampling Technique To Balance the Data

# <h3 style = "color:BlueViolet"><left><i>Down-Sampling of Data to make the data Balanced<i></left></h3>

# In[74]:


df2 = df1.loc[df1['Gender'] == 0]


# In[75]:


df2.head(3)


# In[76]:


df3 = df1.loc[df1['Gender'] == 1]


# In[77]:


df3.head(3)


# In[78]:


df4 = df3.iloc[0:112, :]


# In[79]:


df4.head(3)


# In[80]:


df5 = pd.concat([df2,df4], axis = 0)


# In[81]:


df5


# In[82]:


plt.figure(figsize = (20,4))
plt.suptitle("Difference Between Gender Balance Before Down_Sampling And After Down_Sampling")
plt.subplot(1,2,1)
ax = df1["Gender"].plot.hist()
ax.set_ylabel("Frequecy")
ax.set_xlabel("Gender")
plt.subplot(1,2,2)
ax = df5["Gender"].plot.hist()
ax.set_ylabel("Frequecy")
ax.set_xlabel("Gender")
plt.show()


# <h3 style = "color:BlueViolet"><left><i>Correlation Comparison of Balanced and Imbalanced Data<i></left></h3>

# In[83]:


plt.figure(figsize = (18,9))
plt.suptitle("Correlation Vales Before and After Balanced the data according to Gender")
plt.subplot(1,2,1)
corr = df1.corr()
ax = sns.heatmap(corr, cmap='RdGy', annot=True, linewidths= 1.0)
ax.set_xlabel("When Imnalanced")
plt.subplot(1,2,2)
corr = df5.corr()
ax = sns.heatmap(corr, cmap='RdGy_r', annot=True, linewidths= 1.0)
ax.set_xlabel("When Bnbalanced")
plt.show()


# - From the above information it is clear that Correlation Values improved after balanced the Data

# <h3 style = "color:BlueViolet"><left><i>Feature Scalling<i></left></h3>

# In[84]:


from sklearn.preprocessing import MinMaxScaler


# In[85]:


scale = MinMaxScaler()


# In[86]:


df5["ApplicantIncome"] = scale.fit_transform(np.array(df5["ApplicantIncome"]).reshape(-1,1))


# In[87]:


df5["CoapplicantIncome"] = scale.fit_transform(np.array(df5["CoapplicantIncome"]).reshape(-1,1))


# In[88]:


df5["LoanAmount"] = scale.fit_transform(np.array(df5["LoanAmount"]).reshape(-1,1))


# In[89]:


df5["Loan_Amount_Term"] = scale.fit_transform(np.array(df5["Loan_Amount_Term"]).reshape(-1,1))


# In[90]:


df5["Dependents"] = scale.fit_transform(np.array(df5["Dependents"]).reshape(-1,1))


# In[91]:


df5["Property_Area"] = scale.fit_transform(np.array(df5["Property_Area"]).reshape(-1,1))


# In[92]:


df5


# - After Retuning the features i recognized that the column "Dependents" has no imortance, since i wnat to delecte this feature.

# In[93]:


del df5["Dependents"]


# In[94]:


df5


# <h3 style = "color:BlueViolet"><left><i>Train_Test_Splitting of Data<i></left></h3>

# In[95]:


from sklearn.model_selection import train_test_split


# In[96]:


features = df5.iloc[:,0 : 10]


# In[97]:


x = features
y = df5["Loan_Status"]


# In[98]:


x_train , x_test, y_train, y_test = train_test_split(x, y, random_state = 0, train_size = 0.70, shuffle = True)


# <h3 style = "color:red"><center><i>MACHINE LEARNING MODELS WITH BALANCED DATASETE<i></center></h3>

# <h3 style = "color:Chocolate"><left><i>Logistic Regression<i></left></h3>

# In[99]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression


# In[100]:


model1 = LogisticRegression(random_state = 0)
model1.fit(x_train,y_train)


# In[101]:


# Predicting the Test set results
y_pred = model1.predict(x_test)


# In[102]:


y_pred


# In[103]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred)


# In[104]:


cm1 


# In[105]:


from sklearn.metrics import accuracy_score   # Accuracy score 
ac1 = accuracy_score(y_test, y_pred)


# In[106]:


ac1


# <h3 style = "color:Chocolate"><left><i>K_Nearest_Neighbor(KNN)<i></left></h3>

# In[107]:


# imprt algorithm method name from required libraries
from sklearn.neighbors import KNeighborsClassifier


# In[108]:


# create an alogirthm ( same like as function)
model2 = KNeighborsClassifier(n_neighbors =3)


# In[109]:


# apply the model on training dataset using fit
model2.fit(x_train, y_train)


# In[110]:


# Predicting the Model : input variable of testing datset - xtest
y_pred = model2.predict(x_test)
y_pred


# In[111]:


#Evaluating the Algorithm
from sklearn.metrics import confusion_matrix
# creating confustion matrix table for TP and TN scenarios
cm2 = confusion_matrix(y_test, y_pred)


# In[112]:


cm2


# In[113]:


from sklearn.metrics import accuracy_score
# Calculate the accuracy for the model by validating y_pred and y_test
ac2 = accuracy_score(y_test, y_pred)


# In[114]:


ac2


# <h3 style = "color:Chocolate"><left><i>Support Vector Machine(SVM)<i></left></h3>

# In[115]:


# Importing libraries and Method
from sklearn.svm import SVC
# Creating An alogrithm using imported method names
model3 = SVC(kernel = 'poly')


# In[116]:


# Apply alogrithm on Training dataset (xtrain , ytrain) by using .fitI()
model3.fit(x_train, y_train)


# In[117]:


# Predicting the Test set results by applying on only input variables of
# testind dataset (Xtest)
y_pred = model3.predict(x_test)


# In[118]:


y_pred


# In[119]:


# Making the Confusion Matrix by validating the predicted and Actual Values
# importing required libraries
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test,y_pred)


# In[120]:


cm3


# In[121]:


from sklearn.metrics import accuracy_score
ac3 = accuracy_score(y_test, y_pred)


# In[122]:


ac3


# <h3 style = "color:Chocolate"><left><i>Naive Bayes<i></left></h3>

# In[123]:


from sklearn.naive_bayes import GaussianNB
model4 = GaussianNB()
model4.fit(x_train, y_train)


# In[124]:


y_pred = model4.predict(x_test)


# In[125]:


y_pred


# In[126]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[127]:


# Making the confusion Matrix
cm4 = confusion_matrix(y_test, y_pred)


# In[128]:


cm4


# In[129]:


# Accuracy score
ac4 = accuracy_score(y_test, y_pred)


# In[130]:


ac4


# <h3 style = "color:Chocolate"><left><i>Decision Tree<i></left></h3>

# In[131]:


# Fitting Decision Tree Classification to the Training set
# step 1 :import libraries and methods
from sklearn.tree import DecisionTreeClassifier
# step 2 : Create an alogrithm using imported methods
model5 = DecisionTreeClassifier(criterion = 'entropy')
# Step3 : applying algorithm on training dataset ( Xtrain , ytrain)
model5.fit(x_train, y_train)


# In[132]:


# Predicting the Test set results by applying the algorithm only in input
#variables of testing dataset (xtest)
y_pred = model5.predict(x_test)


# In[133]:


y_pred


# In[134]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test, y_pred)
cm5


# In[135]:


from sklearn.metrics import accuracy_score
ac5 = accuracy_score(y_test, y_pred)


# In[136]:


ac5


# <h3 style = "color:Chocolate"><left><i>Random Forest<i></left></h3>

# In[137]:


# Fitting Random Forest Classification to the Training set
# 1 : Import required libraries and Method for RF
from sklearn.ensemble import RandomForestClassifier
# 2 : create an alogrithm using imported method name
model6 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
# Train the model - Training Data (xtrain, ytrain)
model6.fit(x_train, y_train)


# In[138]:


# Predicting the Test set results - only on input varabiles of dataset
y_pred = model6.predict(x_test)


# In[139]:


y_pred


# In[140]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_test, y_pred)


# In[141]:


cm6


# In[142]:


print(model6.feature_importances_)


# In[143]:


feat_importances = pd.Series(model6.feature_importances_)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[144]:


from sklearn.metrics import accuracy_score
ac6 = accuracy_score(y_test, y_pred)


# In[145]:


ac6


# <h3 style = "color:Chocolate"><left><i>XG BOOST CLASSIFIER<i></left></h3>

# In[146]:


from xgboost import XGBClassifier


# In[147]:


model7 = XGBClassifier()


# In[148]:


model7.fit(x_train,y_train)


# In[149]:


y_pred = model7.predict(x_test)


# In[150]:


y_pred


# In[151]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm7 = confusion_matrix(y_test, y_pred)


# In[152]:


cm7


# In[153]:


from sklearn.metrics import accuracy_score
ac7 = accuracy_score(y_test, y_pred)


# In[154]:


ac7


# <h3 style = "color:Blue"><left><i>Accuracy Scores of ML Algorithms of BALANCED DATA<i></left></h3>

# In[155]:


data = {"Accuracy_Score":[ac1,ac2,ac3,ac4,ac5,ac6,ac7],}
scores1 = pd.DataFrame(data, index = ["LR","KNN","SVM","NB","DT","RF","XGB"])


# In[156]:


scores1


# <h3 style = "color:red"><CENTER><i>MACHINE LAERNING MODELS WITH IMBALANCED DATA<i></CENTER></h3>

# In[157]:


df6 = df1.copy()


# In[158]:


df6


# <h3 style = "color:blueviolet"><left><i>Feature Scalling<i></left></h3>

# In[159]:


from sklearn.preprocessing import MinMaxScaler


# In[160]:


scale = MinMaxScaler()


# In[161]:


df6["ApplicantIncome"] = scale.fit_transform(np.array(df6["ApplicantIncome"]).reshape(-1,1))


# In[162]:


df6["CoapplicantIncome"] = scale.fit_transform(np.array(df6["CoapplicantIncome"]).reshape(-1,1))


# In[163]:


df6["LoanAmount"] = scale.fit_transform(np.array(df6["LoanAmount"]).reshape(-1,1))


# In[164]:


df6["Loan_Amount_Term"] = scale.fit_transform(np.array(df6["Loan_Amount_Term"]).reshape(-1,1))


# In[165]:


df6["Dependents"] = scale.fit_transform(np.array(df6["Dependents"]).reshape(-1,1))


# In[166]:


df6["Property_Area"] = scale.fit_transform(np.array(df6["Property_Area"]).reshape(-1,1))


# In[167]:


df6


# - After Retuning the features i recognized that the column "Dependents" has no imortance, since i wnat to delecte this feature.

# In[168]:


del df6['Dependents']


# In[169]:


df6


# <h3 style = "color:blueviolet"><left><i>Train_Test_Split<i></left></h3>

# In[170]:


from sklearn.model_selection import train_test_split


# In[171]:


features = df6.iloc[:,0:10]


# In[172]:


x = features
y = df6["Loan_Status"]


# In[173]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, train_size = 0.70)


# <h3 style = "color:Chocolate"><left><i>Logistic Regression<i></left></h3>

# In[174]:


from sklearn.linear_model import LogisticRegression
model8 = LogisticRegression(random_state = 0)
model8.fit(x_train,y_train)


# In[175]:


y_pred = model8.predict(x_test)
y_pred


# In[176]:


from sklearn.metrics import confusion_matrix
cm8 = confusion_matrix(y_test, y_pred)


# In[177]:


cm8


# In[178]:


from sklearn.metrics import accuracy_score
ac8 = accuracy_score(y_test,y_pred)


# In[179]:


f1_positive1 = metrics.f1_score(y_test, y_pred, pos_label=1)
f1_positive1


# In[180]:


f1_negative1 = metrics.f1_score(y_test, y_pred, pos_label=0)
f1_negative1


# <h3 style = "color:Chocolate"><left><i>K_Nearest_Neighbor(KNN)<i></left></h3>

# In[181]:


from sklearn.neighbors import KNeighborsClassifier
model9 = KNeighborsClassifier(n_neighbors = 3)
model9.fit(x_train, y_train)


# In[182]:


y_pred = model9.predict(x_test)
y_pred


# In[183]:


from sklearn.metrics import confusion_matrix
cm9 = confusion_matrix(y_test,y_pred)


# cm9

# In[184]:


from sklearn.metrics import accuracy_score
ac9 = accuracy_score(y_test, y_pred)


# In[185]:


ac9


# In[186]:


f1_positive2 = metrics.f1_score(y_test, y_pred, pos_label=1)
f1_positive2


# In[187]:


f1_negative2 = metrics.f1_score(y_test, y_pred, pos_label=0)
f1_negative2 


# <h3 style = "color:Chocolate"><left><i>Support Vector Machine(SVM)<i></left></h3>

# In[188]:


from sklearn.svm import SVC
model10 = SVC(kernel = 'poly')
model10.fit(x_train,y_train)


# In[189]:


y_pred = model10.predict(x_test)
y_pred


# In[190]:


from sklearn.metrics import confusion_matrix
cm10 = confusion_matrix(y_test,y_pred)


# In[191]:


cm10


# In[192]:


from sklearn.metrics import accuracy_score
ac10 = accuracy_score(y_test, y_pred)


# In[193]:


ac10


# In[194]:


f1_positive3 = metrics.f1_score(y_test, y_pred, pos_label=1)
f1_positive3


# In[195]:


f1_negative3 = metrics.f1_score(y_test, y_pred, pos_label=0)
f1_negative3 


# <h3 style = "color:Chocolate"><left><i>Naive Bayes<i></left></h3>

# In[196]:


from sklearn.naive_bayes import GaussianNB
model11 = GaussianNB()
model11.fit(x_train,y_train)


# In[197]:


y_pred = model11.predict(x_test)
y_pred


# In[198]:


from sklearn.metrics import confusion_matrix
cm11 = confusion_matrix(y_test,y_pred)                                                                                                                            


# In[199]:


from sklearn.metrics import accuracy_score
ac11 = accuracy_score(y_test, y_pred)
ac11


# In[200]:


f1_positive4 = metrics.f1_score(y_test, y_pred, pos_label=1)
f1_positive4


# In[201]:


f1_negative4 = metrics.f1_score(y_test, y_pred, pos_label=0)
f1_negative4 


# <h3 style = "color:Chocolate"><left><i>Decision Tree<i></left></h3>

# In[202]:


from sklearn.tree import DecisionTreeClassifier
model12 = DecisionTreeClassifier(criterion = 'entropy')
model12.fit(x_train,y_train)


# In[203]:


y_pred = model12.predict(x_test)
y_pred


# In[204]:


from sklearn.metrics import confusion_matrix
cm12 = confusion_matrix(y_test,y_pred)


# In[205]:


cm12


# In[206]:


from sklearn.metrics import accuracy_score
ac12 = accuracy_score(y_test, y_pred)
ac12


# In[207]:


f1_positive5 = metrics.f1_score(y_test, y_pred, pos_label=1)
f1_positive5


# In[208]:


f1_negative5 = metrics.f1_score(y_test, y_pred, pos_label=0)
f1_negative5 


# <h3 style = "color:Chocolate"><left><i>Random Forest<i></left></h3>

# In[209]:


from sklearn.ensemble import RandomForestClassifier
model13 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
model13.fit(x_train,y_train)


# In[210]:


y_pred = model13.predict(x_test)
y_pred


# In[211]:


from sklearn.metrics import confusion_matrix
cm13 = confusion_matrix(y_test,y_pred)


# In[212]:


cm13


# In[213]:


print(model13.feature_importances_)


# In[214]:


feat_importances_ = pd.Series(model13.feature_importances_)
feat_importances.nlargest(10).plot(kind = 'barh')
plt.show()


# In[215]:


from sklearn.metrics import accuracy_score
ac13 = accuracy_score(y_test,y_pred)
ac13


# In[216]:


f1_positive6 = metrics.f1_score(y_test, y_pred, pos_label=1)
f1_positive6


# In[217]:


f1_negative6 = metrics.f1_score(y_test, y_pred, pos_label=0)
f1_negative6 


# <h3 style = "color:Chocolate"><left><i>X_G_BOOST<i></left></h3>

# In[218]:


from xgboost import XGBClassifier
model14 = XGBClassifier()


# In[219]:


model14.fit(x_train, y_train)


# In[220]:


y_pred = model14.predict(x_test)


# In[221]:


y_pred


# In[222]:


from sklearn.metrics import confusion_matrix
cm14 = confusion_matrix(y_test,y_pred)


# In[223]:


cm14


# In[224]:


from sklearn.metrics import accuracy_score
ac14 = accuracy_score(y_test, y_pred)
ac14


# In[225]:


from sklearn import metrics
f1_positive7 = metrics.f1_score(y_test,y_pred, pos_label=1)
f1_positive7


# In[226]:


f1_negative7 = metrics.f1_score(y_test,y_pred, pos_label=0)
f1_negative7


# <h3 style = "color:Blue"><left><i>F1 Scores and Accuracy Scores of ML Algorithms of IMBALANCED DATA<i></left></h3>

# - Generally F1 Score can be considers to Imbalanced Data.

# In[227]:


data = {"Accuracy_Score":[ac8,ac9,ac10,ac11,ac12,ac13,ac14],
        "F1_Score_Positive":[f1_positive1,f1_positive2,f1_positive3,f1_positive4,f1_positive5,f1_positive6,f1_positive7],
        "F1_Score_Nagative":[f1_negative1,f1_negative2,f1_negative3,f1_negative4,f1_negative5,f1_negative6,f1_negative7]}
scores2 = pd.DataFrame(data, index = ["LR","KNN","SVM","NB","DT","RF","XGB"])


# In[228]:


scores2


# <h3 style = "color:Blue"><left><i>Displaying Classification Evaluation Metrics for Balanced and Imbalanced Data<i></left></h3>

# In[229]:


print("Balanced_Data: ","\n",scores1,"\n","Imbalanced_Data: ","\n",scores2 )


# <h3 style = "color:Blue"><left><i>MODEL SELECTION BASED ON ACCURACY SCORE FOR BALANCED DATASET  AND<i></left></h3>
# <h3 style = "color:Blue"><left><i>F1_SCORES FOR IMBALANCED DATASET<i></left></h3>

# - Frome the above scores, it is clear that: Leniar Regression gives best accuracy score for Balanced dataset and, as well as best f1 score for Imbalanced dataset.
# 
# 
# - Since, based on this i would like to proceed with LinearRegression Model to get the output for test dataset.
# 

# In[230]:


# Loading the test dataset to predict the output

testdata = pd.read_csv(r"C:\\PYTHON\\PANDAS\\test.csv")
testdata


# In[231]:


testdata['Dependents'] = testdata['Dependents'].str.replace("+","")


# In[232]:


# Deleting the Unwanted columns from the data set

del testdata["Loan_ID"]


# In[233]:


del testdata["Dependents"]


# In[234]:


testdata


# <h3 style = "color:BlueViolet"><left><i>Filling Missing Categorical Values with mode()<i></left></h3>

# In[235]:


testdata['Gender'].fillna(testdata.Gender.mode()[0], inplace = True)


# In[236]:


testdata["Married"].fillna(testdata.Married.mode()[0], inplace = True )


# In[237]:


testdata["Self_Employed"].fillna(testdata.Self_Employed.mode()[0], inplace = True )


# In[238]:


testdata["Credit_History"].fillna(testdata.Credit_History.mode()[0], inplace = True )


# In[239]:


testdata["LoanAmount"].fillna(testdata.LoanAmount.mode()[0], inplace = True )


# In[240]:


testdata["Loan_Amount_Term"].fillna(testdata.Loan_Amount_Term.mode()[0], inplace = True )


# In[241]:


testdata.isnull().sum()              # Re_Checking Missing values count from each column After Filled them


# <h3 style = "color:BlueViolet"><left><i>Applying Label Encoding On Categorical Features<i></left></h3>

# In[242]:


from sklearn.preprocessing import LabelEncoder


# In[243]:


encoder = LabelEncoder()


# In[244]:


testdata["Gender"] = encoder.fit_transform(testdata["Gender"])   # Encoding "Gender" Column


# In[245]:


print(testdata["Gender"].unique())                          # Getting unique values of "Gender"


# In[246]:


testdata["Married"] = encoder.fit_transform(testdata["Married"]) # Encoding "Married" Column


# In[247]:


print(testdata["Married"].unique())   


# In[248]:


testdata["Education"] = encoder.fit_transform(testdata["Education"]) # Encoding "Education" column


# In[249]:


print(testdata["Education"].unique())  


# In[250]:


testdata["Self_Employed"] = encoder.fit_transform(testdata["Self_Employed"]) # Encoding "Self_Employed"


# In[251]:


print(testdata["Self_Employed"].unique())   


# In[252]:


testdata["Loan_Amount_Term"] = encoder.fit_transform(testdata["Loan_Amount_Term"]) # Encoding "Loan_Amount_Term"


# In[253]:


print(testdata["Loan_Amount_Term"].unique()) 


# In[254]:


testdata["Property_Area"] = encoder.fit_transform(testdata["Property_Area"]) # Encoding "Property_Area"


# In[255]:


print(testdata["Property_Area"].unique())                               # Getting unique values of "Property_Area"


# In[256]:


testdata.head(3)


# <h3 style = "color:BlueViolet"><left><i>Feature Scalling<i></left></h3>

# In[257]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()


# In[258]:


testdata["ApplicantIncome"] = scale.fit_transform(np.array(testdata["ApplicantIncome"]).reshape(-1,1))


# In[259]:


testdata["CoapplicantIncome"] = scale.fit_transform(np.array(testdata["CoapplicantIncome"]).reshape(-1,1))


# In[260]:


testdata["LoanAmount"] = scale.fit_transform(np.array(testdata["LoanAmount"]).reshape(-1,1))


# In[261]:


testdata["Loan_Amount_Term"] = scale.fit_transform(np.array(testdata["Loan_Amount_Term"]).reshape(-1,1))


# In[262]:


testdata["Property_Area"] = scale.fit_transform(np.array(testdata["Property_Area"]).reshape(-1,1))


# In[263]:


testdata


# <h3 style = "color:Blue"><left><i>Applying Logistic Regression model on test dataset<i></left></h3>

# ##### Here i have choosen logisticregression model to predict the test data, which is applied on Balanced data.

# In[264]:


testdata_pred1 = model1.predict(testdata)


# In[265]:


testdata_pred1


# In[266]:


testdata_pred2 = pd.DataFrame(testdata_pred1)


# ##### Here i have choosen logisticregression model to predict the test data, which is applied on Imbalanced data. 

# In[267]:


testdata_pred3 = model8.predict(testdata)


# In[268]:


testdata_pred3


# In[269]:


testdata_pred4 = pd.DataFrame(testdata_pred3)


# <h3 style = "color:Blue"><left><i>Conclusion :<i></left></h3>

# - All the machine learning models were performed on given train dataset and also models were performed by doing the dataset balanced.
# 
# 
# - For both the balanced and imbalanced datasets retuning methods were done while feature scaling, and applying models, and best steps were kept in.
# 
# 
# - Among all the Machine Learning Classification models, Logistic Regression model gave the best accuracy score for balanced dataset and gave best F1 score for imbalanced dataset.
# 
# 
# - Based on these scores i have choosen Logistic Regression Model for both the balanced and imbalanced datasets to get the output for test dataset.
# 
# ###### Next step is Cross Validation Of Classification ML Algorithms, will be in the next session.
# 
# ###### Thankyou 

# In[ ]:




