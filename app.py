# Core pkgs
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# EDA pkgs
import pandas as pd
import numpy as np

# Data vizualization pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# ML pkgs
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def main():
    '''
        Data visualization app with streamlit
    '''
    st.title('Data Visualization')
    st.caption('Hint: Try using iris.csv')

    activities = ['EDA', 'Plot', 'Model Train']
    choice = st.sidebar.selectbox("Select Activities", activities)

    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")
        
        # Upload
        data = st.file_uploader("Upload Dataset", type=['csv', 'txt'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Show Shape"):
                st.write(df.shape)
            if st.checkbox("Show Columns"):
                all_cols = df.columns.to_list()
                st.write(all_cols)
            if st.checkbox("Select Columns to Display"):
                selected_cols = st.multiselect("Select Columns", all_cols)
                new_df = df[selected_cols]
                st.dataframe(new_df)

            if st.checkbox("Show Summary"):
                st.write(df.describe())
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:, -1].value_counts())
    
    elif choice == 'Plot':
        st.subheader("Data Visualization")

        # Upload
        data = st.file_uploader("Upload Dataset", type=['csv', 'txt'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Correlation with Seaborn"):
                fig, ax = plt.subplots()
                st.write(sns.heatmap(df.corr(), annot=True))
                st.pyplot(fig)

            if st.checkbox("Pie Chart"):
                fig, ax = plt.subplots()
                all_cols = df.columns.to_list()
                cols = st.selectbox('Select a column', all_cols)
                pie_plot = df[cols].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot(fig)
                
            all_cols = df.columns.to_list()
            type_plot = st.selectbox('Select type of plot', ['area', 'bar', 'line', 'hist', 'box', 'kde'])
            selected_cols_name = st.multiselect("Select Columns to plot", all_cols)

            if st.button("Generate Plot"):
                st.success(f'Generating Customizable Plot of {type_plot} for {selected_cols_name}')
                cust_data = df[selected_cols_name]
                # Plot 
                if type_plot == 'area':
                    st.area_chart(cust_data)
                elif type_plot == 'bar':
                    st.bar_chart(cust_data)
                elif type_plot == 'line':
                    st.line_chart(cust_data)
                elif type_plot:
                    # fig, ax = plt.subplots()
                    cust_plot = cust_data.plot(kind=type_plot)
                    st.write(cust_plot)
                    st.pyplot()


    elif choice == 'Model Train':
        st.subheader("Train ML Model")

        # Upload 
        data = st.file_uploader("Upload Dataset", type=['csv', 'txt'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            # Model building
            X = df.iloc[:, 0:-1]
            Y = df.iloc[:, -1]
            seed = 7

            # Model 
            models = []
            models.append(("LR", LogisticRegression()))
            models.append(("LDA", LinearDiscriminantAnalysis()))
            models.append(("KNN", KNeighborsClassifier()))
            models.append(("CART", DecisionTreeClassifier()))
            models.append(("NB", GaussianNB()))
            models.append(("SVM", SVC()))

            # Evaluate Model
            model_name = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'

            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
                cv_result = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                model_name.append(name)
                model_mean.append(cv_result.mean())
                model_std.append(cv_result.std())
                
                accuracy_result = {'model_name': name, 'model_accuracy': cv_result.mean(), 'standard_deviation':cv_result.std()}
                all_models.append(accuracy_result)

            if st.checkbox("Metrics as Table"):
                st.dataframe(pd.DataFrame(zip(model_name, model_mean, model_std), columns=['Model Name', 'Model Accuracy', 'Standard Deviation']))

            if st.checkbox("Metrics as JSON"):
                st.json(all_models)

if __name__ == '__main__':
    main()