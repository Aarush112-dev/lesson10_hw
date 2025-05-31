import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data = pd.read_csv("Lesson 10\successful_educations.csv")
data.info()
data = data.drop(["Graduation Year","GPA (or Equivalent)","Scholarship/Award","University Global Ranking"],axis=1)
data["Degree"].fillna("unknown",inplace=True)
data["Field"].fillna("unknown",inplace=True)
data["Institution"].fillna("unknown",inplace=True)
data.info()


# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split


# x = data[["Degree","Profession","Name","Field"]]
# y=data["Country"].values
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)
# ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),["Degree","Profession","Name","Field"])])
# x = ct.fit_transform(x)

# X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics

# model = DecisionTreeClassifier(max_depth=100,random_state=50,criterion="gini")
# model.fit(X_train,Y_train)
# predictions = model.predict(X_test)
# accuracy = metrics.accuracy_score(predictions,Y_test)
# print(accuracy)

# model = DecisionTreeClassifier(max_depth=100,random_state=50,criterion="entropy")
# model.fit(X_train,Y_train)
# predictions = model.predict(X_test)
# accuracy = metrics.accuracy_score(predictions,Y_test)
# print(accuracy)

#graphs
print(data["Profession"].unique())
data2 = data[data["Profession"]=="Entrepreneur"]
print(data2.shape)
count = data2["Institution"].value_counts().reset_index()
print(count)

figure3 = go.Figure(
    go.Scatter(x=data2["Degree"],y=data["Institution"],mode="markers")
)
figure3.update_layout(title="Institutions of entrepreneurs")
figure3.write_html("int.html",auto_open=True)

# Figure = go.Figure(
#     go.Bar(x=data2[])

# )
