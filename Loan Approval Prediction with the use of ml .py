#!/usr/bin/env python
# coding: utf-8

# # Loan Approval Prediction using Machine Learning  
# 
# ## Introduction  
# 
# Loan approval plays a vital role in the financial sector, where banks and institutions must evaluate whether an applicant is eligible for credit. Traditionally, this decision relies on analyzing multiple factors such as income, employment status, loan amount, credit history, and property details.  
# 
# In this project, I worked with a dataset of loan applications to develop a predictive model that can automatically classify whether a loan should be approved or not. The workflow includes data preprocessing, exploratory data analysis (EDA), visualization, and machine learning model training. The objective is to provide a reliable, data-driven solution that can assist financial institutions in making fair, quick, and accurate loan approval decisions.  
# 
# 
# name - Naitik Sharma 
# 
# 
# 
# email - naitik28sharma@gmail.com
# 
# 
# 
# 
# linkdin - www.linkedin.com/in/naitik-sharma-54627335a

# In[52]:


import pandas as pd

data = pd.read_csv(r"F:\SKILL FOR FUTURE\loan_prediction.csv")
print(data.head())


# In[23]:


df.isnull().sum()


# In[24]:


print(df.describe())


# In[25]:


# Fill missing values in categorical columns with mode
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)


# In[26]:


# Fill missing values in LoanAmount with the median
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Fill missing values in Loan_Amount_Term with the mode
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

# Fill missing values in Credit_History with the mode
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


# In[27]:


import plotly.express as px

loan_status_count = df['Loan_Status'].value_counts()
fig_loan_status = px.pie(loan_status_count, 
                         names=loan_status_count.index, 
                         title='Loan Approval Status')
fig_loan_status.show()


# In[31]:


fig_gender = px.bar(
    df['Gender'].value_counts(),
    title="Gender Distribution",
    text_auto=True,
    color=df['Gender'].value_counts().index,
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig_gender.show()


# In[30]:


fig_married = px.bar(
    df['Married'].value_counts(),
    title="Marital Status Distribution",
    text_auto=True,  # shows values on bars
    color=df['Married'].value_counts().index,  # different colors
    color_discrete_sequence=px.colors.qualitative.Set2  # nice palette
)
fig_married.show()


# In[33]:




fig_education = px.bar(
    df['Education'].value_counts(),
    title="Education Distribution",
    text_auto=True,
    color=df['Education'].value_counts().index,
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig_education.show()


# In[34]:


fig_income = px.histogram(df, x='ApplicantIncome',
                          nbins=40, title='Applicant Income Distribution',
                          color_discrete_sequence=['#4C78A8'], opacity=0.8)
fig_income.update_layout(bargap=0.1, plot_bgcolor='rgba(245,245,245,0.9)',
                         xaxis_title="Applicant Income", yaxis_title="Count")
fig_income.show()


# In[35]:


fig_income = px.box(df, x='Loan_Status', y='ApplicantIncome',
                    color='Loan_Status',
                    title='Loan Status vs Applicant Income',
                    color_discrete_sequence=['#4C78A8', '#E45756'])
fig_income.update_layout(plot_bgcolor='rgba(245,245,245,0.9)',
                         xaxis_title="Loan Status", yaxis_title="Applicant Income")
fig_income.show()


# In[37]:



Q1 = df['ApplicantIncome'].quantile(0.25)
Q3 = df['ApplicantIncome'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['ApplicantIncome'] >= lower_bound) & (df['ApplicantIncome'] <= upper_bound)]


# In[38]:


fig_coapplicant_income = px.box(df, 
                                x='Loan_Status', 
                                y='CoapplicantIncome',
                                color="Loan_Status", 
                                title='Loan_Status vs CoapplicantIncome')
fig_coapplicant_income.show()


# In[39]:


# Calculate the IQR
Q1 = df['CoapplicantIncome'].quantile(0.25)
Q3 = df['CoapplicantIncome'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['CoapplicantIncome'] >= lower_bound) & (df['CoapplicantIncome'] <= upper_bound)]


# In[40]:


fig_loan_amount = px.box(df, x='Loan_Status', 
                         y='LoanAmount', 
                         color="Loan_Status",
                         title='Loan_Status vs LoanAmount')
fig_loan_amount.show()


# In[41]:


fig_credit_history = px.histogram(df, x='Credit_History', color='Loan_Status', 
                                  barmode='group',
                                  title='Loan_Status vs Credit_His')
fig_credit_history.show()


# In[43]:


fig_property_area = px.histogram(df, x='Property_Area', color='Loan_Status', 
                                 barmode='group',
                                title='Loan_Status vs Property_Area')
fig_property_area.show()


# In[44]:


# Convert categorical columns to numerical using one-hot encoding
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
df = pd.get_dummies(df, columns=cat_cols)

# Split the dataset into features (X) and target (y)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical columns using StandardScaler
scaler = StandardScaler()
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

from sklearn.svm import SVC
model = SVC(random_state=42)
model.fit(X_train, y_train)


# In[46]:


#Now let’s make predictions on the test set:
y_pred = model.predict(X_test)
print(y_pred)


# In[47]:


# Convert X_test to a DataFrame
X_test_df = pd.DataFrame(X_test, columns=X_test.columns)

# Add the predicted values to X_test_df
X_test_df['Loan_Status_Predicted'] = y_pred
print(X_test_df.head())


# ## Summary  
# 
# Loan approval prediction focuses on analyzing key factors like an applicant’s financial history, income, credit score, employment status, and other important attributes. By using historical loan data and applying machine learning techniques, we can build models that help predict whether a loan should be approved for new applicants.  
# 
# This project gave me valuable insights into how data preprocessing, feature engineering, and model training all come together to solve real-world problems. I really enjoyed working on this and hope you found it helpful too. If you have any questions or thoughts, feel free to drop them in the comments!  
# 

# In[ ]:




