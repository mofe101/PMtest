import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.colors as mcolors 
import plotly.express as px
import os 
from pathlib import Path 



# print(st.__version__)  

# df1 = pd.read_csv('/Users/mofeogunsola/Documents/Enrollment.csv')
# df2 = pd.read_csv('/Users/mofeogunsola/Documents/Internal Analytics_Benefits and Utilization_Table.csv')


# print(st.__version__)  

# df1 = pd.read_csv('/Users/mofeogunsola/Documents/Enrollment.csv')
# df2 = pd.read_csv('/Users/mofeogunsola/Documents/Internal Analytics_Benefits and Utilization_Table.csv')


# Upload the files
uploaded_file1 = st.file_uploader("Upload Enrollment File", type='csv')
if uploaded_file1 is not None:
    df1 = pd.read_csv(uploaded_file1)
    st.write("Enrollment file loaded:", df1.head())  # Display the first few rows for confirmation

uploaded_file2 = st.file_uploader("Upload Benefits File", type='csv')
if uploaded_file2 is not None:
    df2 = pd.read_csv(uploaded_file2)
    st.write("Benefits file loaded:", df2.head())  # Display the first few rows for confirmation

# Check if the files are loaded before merging
if 'df1' in locals() and 'df2' in locals():
    merged_df = pd.merge(df1, df2, left_on='member_id', right_on='member.member_id')
else:
    st.error("Files not loaded properly, please upload the correct files.")



# Merge 
merged_df = pd.merge(df1, df2, left_on='member_id', right_on='member.member_id')


merged_df = merged_df.drop(columns=['member.member_id'])


#merged_df

df = merged_df

unique_benefit_ids = df['benefit_id'].nunique()
print("Number of unique benefit IDs:", unique_benefit_ids)

benefit_id_mapping = {old_id: f'ID{index + 1}' for index, old_id in enumerate(df['benefit_id'].unique())}

df['benefit_id'] = df['benefit_id'].map(benefit_id_mapping)

# Display the updated dataframe
#print(df)

##print(df.iloc[:, :10])

#unique_merchants = df['Merchant'].dropna().unique()
#print("Unique Merchant Addresses:")

#.for merchant in unique_merchants:
  #  print(merchant)

#from IPython.display import display

df['quarter_of_funding'] = pd.to_datetime(df['quarter_of_funding'], errors='coerce')  # Convert to datetime


def get_quarter(date):
    if pd.isna(date):  # Handle missing or invalid dates
        return None
    month = date.month
    if month in [1, 2, 3]:
        return 'Q1'
    elif month in [4, 5, 6]:
        return 'Q2'
    elif month in [7, 8, 9]:
        return 'Q3'
    else:
        return 'Q4'

df['quarter_of_funding'] = df['quarter_of_funding'].apply(get_quarter)

#display(df)

#dropping mid column cause what is that
df = df.drop(columns=['mid'])#


#df_combined = df_combined.rename(columns={'Spend': 'spend'})

df.rename(columns = {"Spend": "Amount Spent"}, inplace = True)
df.rename(columns = {"quarter_of_funding": "Quarter"}, inplace = True)
df.rename(columns = {"effective_date": "Effective Date"}, inplace = True)
df.rename(columns = {"amount": "Amount Allocated"}, inplace = True)
#df.rename(columns = {"Merchant": "Merchant Address"}, inplace = True)
df.rename(columns = {"Merchant": "Merchant"}, inplace = True)
df.rename(columns = {"benefit_id": "Benefit ID"}, inplace = True)
df.rename(columns = {"member_id": "Member ID"}, inplace = True)
df.rename(columns = {"sponsor_id": "Sponsor ID"}, inplace = True)
df.rename(columns = {"entries.transaction_id": "Entries Transaction ID"}, inplace = True)
df.rename(columns = {"entries.benefit.benefit_id": "Entries Benefit ID"}, inplace = True)

#pd.set_option('display.max_columns', None)
#display(df)


df3 = df[['Member ID', 'Benefit ID', 'Sponsor ID', 'Amount Spent', 
          'Amount Allocated', 'Quarter', 'Effective Date', 
          'Merchant', 'Entries Transaction ID', 
          'Entries Benefit ID']]

#df.reset_index(inplace = True)

df.reset_index(drop=True, inplace=True)  # Reset index first
df.index = df.index + 1

#display(df3)


import re


# Function to clean the Merchant 
def clean_address(address):
    if isinstance(address, str):
        # Strip leading/trailing whitespaces
        address = address.strip()

        # Convert to uppercase (optional)
        address = address.upper()

        # Fix common abbreviations (Optional, you can modify to fit your use case)
        address = re.sub(r'\bST\b', 'STREET', address)  # Abbreviation 'ST' to 'STREET'
        address = re.sub(r'\bAVE\b', 'AVENUE', address)  # Abbreviation 'AVE' to 'AVENUE'
        address = re.sub(r'\bRD\b', 'ROAD', address)  # Abbreviation 'RD' to 'ROAD'
        
        # Remove extra spaces within the address
        address = re.sub(r'\s+', ' ', address)

        # Correct the state code if it's partially inconsistent
        address = re.sub(r'(?<=\w{2})\s*$', '', address)  # Ensure there is a space before state code or make it uniform
        return address
    return address


df['Merchant'] = df['Merchant'].apply(clean_address)

# Show the cleaned column
print(df[['Merchant']].head())

unique_merchants = df['Merchant'].unique() 

benefit_utilization = df.groupby('Benefit ID')[['Amount Spent', 'Amount Allocated']].sum().reset_index()

#Calculate 'Utilization (%)'
benefit_utilization['Utilization (%)'] = (benefit_utilization['Amount Spent'] / benefit_utilization['Amount Allocated']) * 100

#Handle NaN values
benefit_utilization['Utilization (%)'] = benefit_utilization['Utilization (%)'].fillna(0)

#adjust the index to start at 1
benefit_utilization.reset_index(drop=True, inplace=True)
benefit_utilization.index = benefit_utilization.index + 1

df = df.merge(benefit_utilization[['Benefit ID', 'Utilization (%)']], on='Benefit ID', how='left')


df['Utilization (%)'] = df['Utilization (%)'].fillna(0)



# Display the final DataFrame
#display(df)

#display(benefit_utilization)



st.sidebar.image("soda.png")



# Sidebar title and navigation options
st.sidebar.markdown("<h1 style='font-size:30px;'>Benefits Utilization</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='font-size:17px; margin-bottom: 50px;'>Looking into overall Q4 data and merchant impact on utilization rates</h2>", unsafe_allow_html=True)

st.sidebar.title("Dashboard Navigation")
menu = ['Benefit Utilization for Q4', 'Utilization Rates for San Francisco Merchants', 'Dataset']
choice = st.sidebar.selectbox('Select a Page:', menu)

# Page 1: Benefit Utilization Graph for Q4
if choice == 'Benefit Utilization for Q4':
    st.subheader('Benefit Utilization Rates for Q4')

  #figures?
    plt.figure(figsize=(12, 9.5))

    barplot = sns.barplot(x='Benefit ID', y='Utilization (%)', data=benefit_utilization, palette='viridis')

    plt.xlabel('Benefit ID', fontsize=16, fontweight='bold')
    plt.ylabel('Utilization Rate (%)', fontsize=16, fontweight='bold')

    #
    norm = mcolors.Normalize(vmin=benefit_utilization['Utilization (%)'].min(), vmax=benefit_utilization['Utilization (%)'].max())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca())

    # Add utilization % on top of each bar
    for p in barplot.patches:
        height = p.get_height()
        barplot.text(p.get_x() + p.get_width() / 2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=12)

    # Show plot
    st.pyplot(plt)

# Page 2: Utilization Rates for San Francisco Merchants
elif choice == 'Utilization Rates for San Francisco Merchants':
    st.subheader('Utilization Rates for San Francisco Merchants')

    # Filter out specific merchants
    merchant_list = [
        "SAFEWAY #1490 SAN FRANCISCO CA",
        "CVS/PHARMACY #04675 SAN FRANCISCO CA",
        "WALGREENS #4570 SAN FRANCISCO CA",
        "FOODSCO #0351 SAN FRANCISCO CA"
    ]
    
    # Filter data
    df_filtered = df[df['Merchant'].isin(merchant_list)]

 
    avg_utilization = df_filtered.groupby('Merchant')['Utilization (%)'].mean().sort_values()

 
    avg_utilization_df = avg_utilization.reset_index()


    fig = px.bar(
        avg_utilization_df,
        x="Merchant", 
        y="Utilization (%)",
        labels={"Utilization (%)": "Utilization Rate (%)", "Merchant": ""},
        #title="Average Utilization Rate by Merchant in San Francisco",

    )

    #layout
    fig.update_layout(
        yaxis=dict(range=[25, 33]),  
        font=dict(size=14),  # General font size
        hoverlabel=dict(font_size=14),
          width=900,  # Set the width of the figure
        height=700,
        showlegend=False  
    )

    st.plotly_chart(fig)

# Page 3: View Dataset
elif choice == 'Dataset':
    st.subheader('Full Dataset')
    st.dataframe(df)


    st.dataframe(df, height=1000)
