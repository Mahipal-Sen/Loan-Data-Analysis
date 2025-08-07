import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




# 1. Data Handling
# Load the dataset using Pandas.


df=pd.read_csv( 'train_indessa.csv')
# Inspect the dataset:
# View the shape, column names, and .
print(df,'\n')
print('SHAPE OF THE DATA:','\n')
print(df.shape),'\n'

# print(df)
# View the data types
print('INFORMATION OF THE DATA:','\n')
df.info(),'\n'

print('DISCRIPTION OF THE DATA;','\n')
print(df.describe(),'\n')
# Identify missing values:

print('MISSINFG VALUES OF THE coloums:','\n')
null_cols = df.columns[df.isnull().any()]
print(null_cols),'\n'

# Count missing entries in each column and Calculate the percentage of missing data per column.:

print('COUNT OF MISSING VALUE PER COLUMN AND THEIR PERCENTAGE:','\n')

null_df = df[null_cols].isnull().sum().to_frame(name='Null Count')\
          .merge(df[null_cols].isnull().mean().mul(100).to_frame(name='Null Percent'), left_index=True, right_index=True)

null_df_sorted = null_df.sort_values(by='Null Count', ascending=False)


print(null_df_sorted),'\n'

df[df.duplicated()].shape[0]

df.drop(['verification_status_joint','desc','mths_since_last_record','mths_since_last_major_derog','mths_since_last_delinq'], axis=1, 
        inplace=True)

df_new = df[['funded_amnt_inv', 'term', 'int_rate', 'emp_length', 'emp_title', 'annual_inc',
             'verification_status', 'purpose', 'addr_state', 'dti', 'initial_list_status',
             'total_rec_int', 'total_rec_late_fee', 'application_type', 'tot_coll_amt',
             'tot_cur_bal', 'loan_status']].rename(columns={
    'funded_amnt_inv': 'BANK_INVESTMENT',
    'term': 'TERM',
    'int_rate': 'INTEREST_RATE',
    'emp_length': 'EMPLOYEMENT_DURATION',
    'emp_title': 'EMPPLOYMENT_TITLE',
    'annual_inc': 'ANNUAL_INCOME',
    'verification_status': 'STATUS',
    'purpose': 'LOAN_PURPOSE',
    'addr_state': 'STATE',
    'dti': 'DTI',
    'initial_list_status': 'INITIAL_LIST_STATUS',
    'total_rec_int': 'RECEIVED_INTEREST_TOTAL',
    'total_rec_late_fee': 'RECEIVED_LATE_FEE',
    'application_type': 'APPLICATION_TYPE',
    'tot_coll_amt': 'TOTAL_COLLECTION_AMOUNT',
    'tot_cur_bal': 'TOTAL_CURRENT_BALANCE',
    'loan_status': 'LOAN_STATUS'
})
# new file

df_new.to_csv('EDA_data.csv', index=False)
df= pd.read_csv('EDA_data.csv')
print(df.shape),'\n' 

print('INFORMATION OF THE UPDATED DATA:','\n')
df.info(),'\n'


df.head()


#RECHECK FOR NULL VALUES WITH COUNT
print('RECHECKED NULL VALUES WITH COUNT:','\n')
null_cols = df.columns[df.isnull().any()]
null_df = df[null_cols].isnull().sum().to_frame(name='Null Count')\
          .merge(df[null_cols].isnull().mean().mul(100).to_frame(name='Null Percent'), left_index=True, right_index=True)
null_df_sorted = null_df.sort_values(by='Null Count', ascending=False)
print(null_df_sorted,'\n')


distinct_entries = pd.Series(df['EMPPLOYMENT_TITLE'].value_counts()).sort_values(ascending=False)
# print sorted unique values
print(distinct_entries)

#drop the column
df.drop(['EMPPLOYMENT_TITLE'], axis=1, inplace=True)

#CHANGING THE DATATYPE
df['TERM'] = df['TERM'].str.replace('months', '')
df['TERM'] = df['TERM'].astype(int)
df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].str.replace('years', '')
df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].str.replace('year', '')
df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].str.replace('+', '')    # In our analysis we will consider 10 as 10+ years 
df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].str.replace('< 1', '0') # In our analysis we will consider 0 as less than a year
df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].fillna('-1')
df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].astype(int)

#CHECK FOR NULL VALUES WITH COUNT
null_cols = df.columns[df.isnull().any()]
null_df = df[null_cols].isnull().sum().to_frame(name='Null Count')\
          .merge(df[null_cols].isnull().mean().mul(100).to_frame(name='Null Percent'), left_index=True, right_index=True)
null_df_sorted = null_df.sort_values(by='Null Count', ascending=False)
print(null_df_sorted)

df.to_csv('EDA_ready_data.csv', index=False)

df= pd.read_csv('EDA_ready_data.csv')

# separated categorical variables for easy analysis.

column_category=df.select_dtypes(include=['object']).columns
print(column_category)

# separated numerical variables for easy analysis.
column_numerical = df.select_dtypes(include=np.number).columns.tolist()
print(column_numerical)

# A statistics summary of ALL data

print(df.describe(include='all').T,'\n')

print(df.describe().T,'\n')

# analysis using Histogram and Box Plot for continuous variables like BANK_INVESTMENT and INTEREST_RATE
plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(x=df['INTEREST_RATE'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of BANK_INVESTMENT')
plt.xlabel('INTEREST_RATE')
plt.ylabel('Count')

# Boxplot
plt.subplot(1, 2, 2)
plt.boxplot(df['INTEREST_RATE'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Boxplot of BANK_INVESTMENT')
plt.xlabel('INTEREST_RATE')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Bar Plot for Categorical Variables', fontsize=16, ) 

# 1. STATUS
status_counts = df['STATUS'].value_counts()
axes[0, 0].bar(status_counts.index, status_counts.values, color='purple')
axes[0, 0].set_title('STATUS')
axes[0, 0].set_xlabel('Status')
axes[0, 0].set_ylabel('Count')

# 2. LOAN_PURPOSE (top 5 only)
purpose_counts = df['LOAN_PURPOSE'].value_counts().head(5)
axes[0, 1].bar(purpose_counts.index, purpose_counts.values, color='purple',y=1.02)
axes[0, 1].set_title('LOAN_PURPOSE (Top 5)')
axes[0, 1].set_xlabel('Loan Purpose')
axes[0, 1].set_ylabel('Count')
axes[0, 1].tick_params(axis='x', rotation=10,)

# 3. STATE (top 20 only)
state_counts = df['STATE'].value_counts().head(20)
axes[1, 0].bar(state_counts.index, state_counts.values, color='purple')
axes[1, 0].set_title('STATE (Top 20)')
axes[1, 0].set_xlabel('State')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. INITIAL_LIST_STATUS
list_status_counts = df['INITIAL_LIST_STATUS'].value_counts()
axes[1, 1].bar(list_status_counts.index, list_status_counts.values, color='purple',y=-1.02)
axes[1, 1].set_title('INITIAL_LIST_STATUS')
axes[1, 1].set_xlabel('Initial List Status')
axes[1, 1].set_ylabel('Count')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96]) # Adjust layout to fit suptitle
plt.savefig('ploat1.png')
plt.show()

# TO CHECK THE LINK BETWEEN DATA MAKE SCATTER PLOT

sns.pairplot(df[column_numerical])
plt.savefig('pairplot.png')
plt.show()

fig, axarr = plt.subplots(3, 2, figsize=(8, 10))  # Bigger width
plt.subplots_adjust(hspace=0.5, wspace=0.4)
df.groupby('STATUS')['BANK_INVESTMENT'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][0], fontsize=12,color='purple')
axarr[0][0].set_title("Status Vs Bank Investment(Average)", fontsize=18)
axarr[0][0].set_xticklabels(axarr[0][0].get_xticklabels(), rotation=0)

df.groupby('LOAN_PURPOSE')['BANK_INVESTMENT'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][1], fontsize=7,color='purple',y=1.02)
axarr[0][1].set_title("Loan Purpose Vs Bank Investment(Average)", fontsize=12)
axarr[0][1].tick_params(axis='x', rotation=25)

df.groupby('STATE')['BANK_INVESTMENT'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[1][0], fontsize=12,color='purple',)
axarr[1][0].set_title("State vs top 5 States by Bank Investment(Average)", fontsize=12)
axarr[1][0].set_xticklabels(axarr[1][0].get_xticklabels(), rotation=0)

df.groupby('INITIAL_LIST_STATUS')['BANK_INVESTMENT'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][1], fontsize=12,color='purple',y=-1.02)
axarr[1][1].set_title("Initial list status Vs Bank Investment(Average)", fontsize=12)
axarr[1][1].set_xticklabels(axarr[1][1].get_xticklabels(), rotation=0)

df.groupby('APPLICATION_TYPE')['BANK_INVESTMENT'].mean().sort_values(ascending=True).head(10).plot.bar(ax=axarr[2][0], fontsize=8,color='purple')
axarr[2][0].set_title("Application type Vs Bank Investment(Average)", fontsize=12)
axarr[2][0].set_xticklabels(axarr[2][0].get_xticklabels(), rotation=0)

df.groupby('TERM')['BANK_INVESTMENT'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[2][1], fontsize=12,color='purple')
axarr[2][1].set_title("Term Vs Bank Investment(Average) ", fontsize=12)
axarr[2][1].set_xticklabels(axarr[2][1].get_xticklabels(), rotation=0)

plt.subplots_adjust(hspace=1.0)
plt.subplots_adjust(wspace=.5)
sns.despine()
plt.savefig('ploat2.png')
plt.tight_layout()
plt.show()

# to study multiple variable relationship
plt.figure(figsize=(12, 7))
numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
sns.heatmap(numeric_df.corr(), annot=True, vmin=-1, vmax=1, cmap="Purples")
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()