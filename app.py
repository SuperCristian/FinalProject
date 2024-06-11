import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------Config
st.set_page_config(page_title="Obesity Category", page_icon=":bar_chart:", layout="wide")

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Title of the app
st.title(":bar_chart: Simple Obesity Category Site")
st.write("This is a simple website for the Final Project in ITC130. These are the codes on GitHub 👉[Github Link](https://www.kaggle.com/datasets/mrsimple07/obesity-prediction/data) ")


# Load your dataset
url = 'https://drive.google.com/uc?export=download&id=1oqeQp6QwkaWscEek68eoYMvmQsWrhZ8t'
df = pd.read_csv(url) 

# Sidebar for dataset information and feature selection
st.sidebar.header("Dataset Information")
st.sidebar.write("Obesity Category Dataset by Kaggle 👇")
st.sidebar.markdown('''
__This is the source of the dataset and problem statement__ [Kaggle Link](https://www.kaggle.com/datasets/mrsimple07/obesity-prediction/data)  
The dataset consists of 1000 records of data, 7 features + target  
''')

st.sidebar.header("Select Feature")
feature = st.sidebar.selectbox("Choose a feature", df.columns)

# Sidebar for additional options
st.sidebar.header("Additional Options")
show_raw_data = st.sidebar.checkbox("Show raw data")
show_statistics = st.sidebar.checkbox("Show statistics")

# Sidebar for age range
st.sidebar.header("Filter by Age")
min_age = st.sidebar.number_input("Minimum Age", min_value=0, max_value=150, value=0, step=1)
max_age = st.sidebar.number_input("Maximum Age", min_value=0, max_value=150, value=100, step=1)

# Filter data by age range
filtered_df = df[(df['Age'] >= min_age) & (df['Age'] <= max_age)]

# Sidebar for donut chart parameter
st.sidebar.subheader('Donut chart parameter')
donut_theta = st.sidebar.selectbox('Select data', df.select_dtypes(include=['object']).columns)

# Sidebar for line chart parameters
st.sidebar.subheader('Line chart parameters')
plot_data = st.sidebar.multiselect('Select data', df.select_dtypes(include=['float64', 'int64']).columns, df.select_dtypes(include=['float64', 'int64']).columns.tolist())
plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created by ***Cristian Jaycob Estalane*** and ***Larry Diano***
___With the Help of ChatGPT___
''')

# Main content

# Columns for layout
col1, col2 = st.columns(2)

# Row A: Metrics
with col1:
    st.markdown('### Metrics')
    col1.metric("Average Age", f"{df['Age'].mean():.1f} years old")
    col1.metric("Average Height", f"{df['Height'].mean():.1f} cm")
    col1.metric("Average Weight", f"{df['Weight'].mean():.1f} kg")

# Row B: Donut chart
with col2:
    st.markdown('### Donut chart')
    fig, ax = plt.subplots()
    obesity_counts = filtered_df[donut_theta].value_counts()
    ax.pie(obesity_counts, labels=obesity_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# Row C: Line chart
st.markdown('### Line chart')
fig, ax = plt.subplots(figsize=(10, 5))
for data in plot_data:
    sns.lineplot(x='Age', y=data, data=filtered_df, ax=ax, label=data)
ax.set_ylabel("Value")
ax.set_title("Line Chart")
st.pyplot(fig)

# Plotting
st.header(f"Distribution and Box Plot of {feature}")

# Columns for plots
col1, col2 = st.columns(2)

# Distribution plot
with col1:
    st.subheader("Distribution Plot")
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

# Box plot
with col2:
    st.subheader("Box Plot")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[feature], ax=ax)
    st.pyplot(fig)

# Show statistics
if show_statistics:
    st.header(f"Statistics of {feature}")
    st.write(df[feature].describe())

# Display raw data
if show_raw_data:
    st.header("Raw Data")
    st.write(df)

# Relationship between Age and Obesity Category
st.header("Relationship Between Age and Obesity Category")

# Swarm plot
fig, ax = plt.subplots()
sns.swarmplot(x='Age', y='ObesityCategory', data=filtered_df, ax=ax)
plt.xlabel('Age')
plt.ylabel('Obesity Category')
plt.title('Age vs. Obesity Category')
st.pyplot(fig)
