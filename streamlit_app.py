import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 🔹 Streamlit must be configured first!
st.set_page_config(page_title="Student Performance Analysis", layout="wide")

# 🔹 Load Dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('StudentsPerformance.csv')
        data['total_score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)
        data['percentage'] = data['total_score']
        data['pass_fail'] = np.where(data['total_score'] >= 50, 'Pass', 'Fail')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_data()

# 🔹 Streamlit UI with Styling
st.title("📊 **Student Performance Prediction and Analysis**")
st.markdown("<style>div.block-container {padding-top: 2rem;} h1 {color: #2E86C1;} .stSlider {background-color: #F4F6F7;}</style>", unsafe_allow_html=True)

# 🔹 Tabs Navigation
tabs = ["Home", "Context", "Analysis", "Predictions", "Recommendations"]
selected_tab = st.radio("**Navigation**", tabs, index=0, key="tabs")

# 🔹 Home Page
if selected_tab == "Home":
    st.header("🏠 **Welcome to the Student Performance Analysis Tool**")
    st.markdown("""
    **📌 Features of this App:**
    - **Context:** Understand the dataset.
    - **Analysis:** Visual insights into student performance.
    - **Predictions:** ML model predicts student results.
    - **Recommendations:** AI-powered study suggestions.
    """)

# 🔹 Context Tab
elif selected_tab == "Context":
    st.header("📂 **Dataset Overview**")
    if not data.empty:
        st.dataframe(data.head())
    else:
        st.warning("No data available.")

# 🔹 Analysis Tab
elif selected_tab == "Analysis":
    st.header("📊 **Data Visualizations**")
    if not data.empty:
        col1, col2 = st.columns(2)

        with col1:
            visualization_type = st.selectbox("**Select Visualization Type**", [
                "Score Distribution", "Pass/Fail Analysis", "Scatter Plots", "Correlation Heatmap", "Pair Plot"
            ])

        with col2:
            st.write("## ")

        if visualization_type == "Score Distribution":
            fig = px.histogram(data, x='total_score', color='pass_fail', marginal='box', title='Student Score Distribution')
            st.plotly_chart(fig)

        elif visualization_type == "Pass/Fail Analysis":
            pass_fail_count = data['pass_fail'].value_counts().reset_index()
            pass_fail_count.columns = ['Result', 'Count']
            fig = px.bar(pass_fail_count, x='Result', y='Count', color='Result', color_discrete_map={'Pass': 'green', 'Fail': 'red'})
            st.plotly_chart(fig)

        elif visualization_type == "Scatter Plots":
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            sns.scatterplot(data=data, x='math score', y='reading score', hue='pass_fail', ax=ax[0])
            ax[0].set_title("Math vs Reading Scores")
            sns.scatterplot(data=data, x='math score', y='writing score', hue='pass_fail', ax=ax[1])
            ax[1].set_title("Math vs Writing Scores")
            sns.scatterplot(data=data, x='reading score', y='writing score', hue='pass_fail', ax=ax[2])
            ax[2].set_title("Reading vs Writing Scores")
            st.pyplot(fig)

        elif visualization_type == "Correlation Heatmap":
            numeric_data = data.select_dtypes(include=[np.number])
            fig, ax = plt.subplots()
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        elif visualization_type == "Pair Plot":
            pair_fig = sns.pairplot(data, hue='pass_fail')
            st.pyplot(pair_fig)
    else:
        st.warning("No data available for analysis.")

# 🔹 Predictions Tab
elif selected_tab == "Predictions":
    st.header("🎯 **Student Performance Predictions**")
    if not data.empty:
        model_type = st.radio("**Select Prediction Type**", ["Classification", "Regression"])

        if model_type == "Classification":
            X = data[['math score', 'reading score', 'writing score']]
            y = data['pass_fail']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write(f"**Model Accuracy:** `{accuracy_score(y_test, y_pred):.2f}`")
            st.text(classification_report(y_test, y_pred))

        elif model_type == "Regression":
            X = data[['math score', 'reading score', 'writing score']]
            y = data['total_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write(f"**Mean Absolute Error:** `{mean_absolute_error(y_test, y_pred):.2f}`")
            st.write(f"**R² Score:** `{r2_score(y_test, y_pred):.2f}`")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color="blue")
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
            ax.set_xlabel("Actual Scores")
            ax.set_ylabel("Predicted Scores")
            ax.set_title("Actual vs Predicted Performance")
            st.pyplot(fig)

# 🔹 Recommendations Tab
elif selected_tab == "Recommendations":
    st.header("🤖 **AI-Powered Personalized Recommendations**")
    percentage = st.slider("Select Your Percentage", 0, 100, 50, 1)

    if percentage < 30:
        st.write("🚨 **Focus on strengthening your basics.**")
    elif percentage < 50:
        st.write("📈 **You're improving! Focus on problem-solving and time management.**")
    elif percentage < 85:
        st.write("🔥 **You're doing well! Focus on mastering core concepts.**")
    else:
        st.write("🏆 **Excellent! Aim for advanced problem-solving and leadership skills.**")
