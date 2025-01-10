import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import re

# Print Streamlit version
st.write(f"Streamlit Version: {st.__version__}")

# Placeholder for file upload widgets
upload_placeholder = st.empty()
file1 = upload_placeholder.file_uploader("Upload Enrollment.csv", type="csv")
file2 = upload_placeholder.file_uploader("Upload Internal Analytics CSV", type="csv")

# Logic for file uploads and conditional display
if file1 is not None and file2 is not None:
    # Clear the file upload widgets
    upload_placeholder.empty()

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Button to control display
    if st.button('Show Raw Data'):
        st.subheader("Enrollment CSV:")
        st.write(df1.head())
        st.subheader("Internal Analytics CSV:")
        st.write(df2.head())
else:
    st.warning("Please upload both CSV files.")

# Merge datasets
merged_df = pd.merge(df1, df2, left_on='member_id', right_on='member.member_id')
merged_df = merged_df.drop(columns=['member.member_id'])
df = merged_df

# Process data
unique_benefit_ids = df['benefit_id'].nunique()
benefit_id_mapping = {old_id: f'ID{index + 1}' for index, old_id in enumerate(df['benefit_id'].unique())}
df['benefit_id'] = df['benefit_id'].map(benefit_id_mapping)

def get_quarter(date):
    if pd.isna(date):
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

df['quarter_of_funding'] = pd.to_datetime(df['quarter_of_funding'], errors='coerce').apply(get_quarter)
df = df.drop(columns=['mid'])
df.rename(columns={
    "Spend": "Amount Spent",
    "quarter_of_funding": "Quarter",
    "effective_date": "Effective Date",
    "amount": "Amount Allocated",
    "Merchant": "Merchant",
    "benefit_id": "Benefit ID",
    "member_id": "Member ID",
    "sponsor_id": "Sponsor ID",
    "entries.transaction_id": "Entries Transaction ID",
    "entries.benefit.benefit_id": "Entries Benefit ID"
}, inplace=True)

def clean_address(address):
    if isinstance(address, str):
        address = address.strip().upper()
        address = re.sub(r'\bST\b', 'STREET', address)
        address = re.sub(r'\bAVE\b', 'AVENUE', address)
        address = re.sub(r'\bRD\b', 'ROAD', address)
        address = re.sub(r'\s+', ' ', address)
        return address
    return address

df['Merchant'] = df['Merchant'].apply(clean_address)

# Calculate benefit utilization
benefit_utilization = df.groupby('Benefit ID')[['Amount Spent', 'Amount Allocated']].sum().reset_index()
benefit_utilization['Utilization (%)'] = (benefit_utilization['Amount Spent'] / benefit_utilization['Amount Allocated']) * 100
benefit_utilization['Utilization (%)'] = benefit_utilization['Utilization (%)'].fillna(0)
benefit_utilization.reset_index(drop=True, inplace=True)
benefit_utilization.index = benefit_utilization.index + 1
df = df.merge(benefit_utilization[['Benefit ID', 'Utilization (%)']], on='Benefit ID', how='left')
df['Utilization (%)'] = df['Utilization (%)'].fillna(0)

# Sidebar
st.sidebar.image("soda.png")
st.sidebar.markdown("<h1 style='font-size:30px;'>Benefits Utilization</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='font-size:17px; margin-bottom: 50px;'>Looking into overall Q4 data and merchant impact on utilization rates</h2>", unsafe_allow_html=True)
st.sidebar.title("Dashboard Navigation")
menu = ['Benefit Utilization for Q4', 'Utilization Rates for San Francisco Merchants', 'Dataset']
choice = st.sidebar.selectbox('Select a Page:', menu)

# Benefit Utilization for Q4
if choice == 'Benefit Utilization for Q4':
    st.subheader('Benefit Utilization Rates for Q4')
    plt.figure(figsize=(12, 9.5))
    barplot = sns.barplot(x='Benefit ID', y='Utilization (%)', data=benefit_utilization, palette='viridis')
    plt.xlabel('Benefit ID', fontsize=16, fontweight='bold')
    plt.ylabel('Utilization Rate (%)', fontsize=16, fontweight='bold')
    norm = mcolors.Normalize(vmin=benefit_utilization['Utilization (%)'].min(), vmax=benefit_utilization['Utilization (%)'].max())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca())
    for p in barplot.patches:
        height = p.get_height()
        barplot.text(p.get_x() + p.get_width() / 2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
    st.pyplot(plt)

# Utilization Rates for San Francisco Merchants
elif choice == 'Utilization Rates for San Francisco Merchants':
    st.subheader('Utilization Rates for San Francisco Merchants')
    merchant_list = [
        "SAFEWAY #1490 SAN FRANCISCO CA",
        "CVS/PHARMACY #04675 SAN FRANCISCO CA",
        "WALGREENS #4570 SAN FRANCISCO CA",
        "FOODSCO #0351 SAN FRANCISCO CA"
    ]
    df_filtered = df[df['Merchant'].isin(merchant_list)]
    avg_utilization = df_filtered.groupby('Merchant')['Utilization (%)'].mean().sort_values()
    avg_utilization_df = avg_utilization.reset_index()
    fig = px.bar(
        avg_utilization_df,
        x="Merchant", 
        y="Utilization (%)",
        labels={"Utilization (%)": "Utilization Rate (%)", "Merchant": ""},
    )
    fig.update_layout(
        yaxis=dict(range=[25, 33]),  
        font=dict(size=14),  
        hoverlabel=dict(font_size=14),
        width=900,  
        height=700,
        showlegend=False
    )
    st.plotly_chart(fig)

# View Dataset
elif choice == 'Dataset':
    st.subheader('Full Dataset')
    st.dataframe(df)

