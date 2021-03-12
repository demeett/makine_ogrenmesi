import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
meta_df=pd.read_csv(r'C:\Users\demir\Downloads\ISIC_2019_Training_Metadata (1).csv')
ground_df=pd.read_csv(r'C:\Users\demir\Downloads\ISIC_2019_Training_GroundTruth (1).csv')
meta_df.drop("lesion_id",axis=1,inplace=True)
print(meta_df.head)

meta_df=meta_df.dropna(axis=0)
print(meta_df.isnull().sum())
print(meta_df.shape)
print(meta_df.head)
meta_df["sınıf"]=None
for i in range(len(meta_df)):
    for kolon in ground_df.columns:
        if(ground_df[kolon][i]==True):
            meta_df["sınıf"][i]=kolon


meta_df.to_csv(r'ödev.csv', index = False)
ödev_df=pd.read_csv(r'C:\Users\demir\ödev.csv')
print(ödev_df.head)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
meta_df=pd.read_csv(r'C:\Users\demir\Downloads\ISIC_2019_Training_Metadata (1).csv')
ground_df=pd.read_csv(r'C:\Users\demir\Downloads\ISIC_2019_Training_GroundTruth (1).csv')
meta_df.drop("lesion_id",axis=1,inplace=True)
print(meta_df.head)

meta_df=meta_df.dropna(axis=0)
print(meta_df.isnull().sum())
print(meta_df.shape)
print(meta_df.head)
meta_df["sınıf"]=None
for i in range(len(meta_df)):
    for kolon in ground_df.columns:
        if(ground_df[kolon][i]==True):
            meta_df["sınıf"][i]=kolon


meta_df.to_csv(r'ödev.csv', index = False)
ödev_df=pd.read_csv(r'C:\Users\demir\ödev.csv')
ödev_df=pd.read_csv(r'C:\Users\demir\ödev.csv')
ödev_df = ödev_df.dropna(axis=0)
print (ödev_df.isnull().sum())
print (ödev_df.shape)
#meta_df.gender=meta_df.gender.astype(int)
ödev_df.age_approx=ödev_df.age_approx.astype(int)
#meta_df.rc=meta_df.rc.astype(float)
print (ödev_df.info())
print (ödev_df.head(80))
dtype_object=ödev_df.select_dtypes(include=['object'])
dtype_object.head()
for x in dtype_object.columns:
    print("{} unique values:".format(x),ödev_df[x].unique())
    print("*"*20)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dtype_object=ödev_df.select_dtypes(include=['object'])
print (dtype_object.head())
for x in dtype_object.columns:
    ödev_df[x]=le.fit_transform(ödev_df[x])

print (ödev_df.head())
X = ödev_df.iloc[:,:4].values
y = ödev_df['sınıf'].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)
# Feature Scaling to bring the variable in a single scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
score=[]
algorithms=[]
#KNN
from sklearn.neighbors import KNeighborsClassifier
#model and accuracy
knn=KNeighborsClassifier(n_neighbors=33)
knn.fit(X_train,y_train)
knn.predict(X_test)
score.append(knn.score(X_test,y_test)*100)
algorithms.append("KNN")
print("KNN accuracy =",knn.score(X_test,y_test)*100)
#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)
#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title(" KNN Confusion Matrix")
plt.show()
from sklearn.metrics import classification_report
target_names=['NV', 'MEL', 'BKL', 'DF', 'SCC', 'BCC', 'VASC', 'AK']
print(classification_report(y_true, y_pred, target_names=target_names))
print(ödev_df.head)
#Navie-Bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
#Training
nb.fit(X_train,y_train)
#Test
score.append(nb.score(X_test,y_test)*100)
algorithms.append("Navie-Bayes")
print("Navie Bayes accuracy =",nb.score(X_test,y_test)*100)
from sklearn.metrics import confusion_matrix
y_pred=nb.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Navie Bayes Confusion Matrix")
plt.show()
target_names=['NV', 'MEL', 'BKL', 'DF', 'SCC', 'BCC', 'VASC', 'AK']
print(classification_report(y_true, y_pred, target_names=target_names))
#Support Vector Machine
from sklearn.svm import SVC
svm=SVC(random_state=1)
svm.fit(X_train,y_train)
score.append(svm.score(X_test,y_test)*100)
algorithms.append("Support Vector Machine")
print("svm test accuracy =",svm.score(X_test,y_test)*100)
#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=svm.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Support Vector Machine Confusion Matrix")
plt.show()
target_names=['NV', 'MEL', 'BKL', 'DF', 'SCC', 'BCC', 'VASC', 'AK']
print(classification_report(y_true, y_pred, target_names=target_names))
# DecisionTree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
print("Decision Tree accuracy:",dt.score(X_test,y_test)*100)
score.append(dt.score(X_test,y_test)*100)
algorithms.append("Decision Tree")
#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=dt.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Decision Tree Confusion Matrix")
plt.show()
target_names=['NV', 'MEL', 'BKL', 'DF', 'SCC', 'BCC', 'VASC', 'AK']
print(classification_report(y_true, y_pred, target_names=target_names))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)
score.append(lr.score(X_test,y_test)*100)
algorithms.append("Logistic Regression")
print("Logistic Regression accuracy {}".format(lr.score(X_test,y_test)*100))
from sklearn.metrics import confusion_matrix
y_pred=lr.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Logistic Regression Confusion Matrix")
plt.show()
target_names=['NV', 'MEL', 'BKL', 'DF', 'SCC', 'BCC', 'VASC', 'AK']
print(classification_report(y_true, y_pred, target_names=target_names))
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#df=pd.read_csv("data.csv")

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()

# butun kolonlardaki unik degerleri gosterir
dtype_object=ödev_df.select_dtypes(include=['object'])
print (dtype_object.head())
for x in dtype_object.columns:
    ödev_df[x]=le.fit_transform(ödev_df[x])

print (ödev_df.head())
X = ödev_df.iloc[:,:4].values
print X.shape[0]
y = df['sınıf'].values.reshape(X.shape[0], 1)
#split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=82)

#standardize the dataset
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
sc.fit(X_test)
X_test = sc.transform(X_test)

sknet = MLPClassifier(hidden_layer_sizes=(8), learning_rate_init=0.01, max_iter=100)
sknet.fit(X_train, y_train)

score.append(sknet.score(X_test,y_test)*100)
algorithms.append("Artificial Neural Networks")
print("Artificial Neural Networks accuracy {}".format(sknet.score(X_test,y_test)*100))

y_pred = sknet.predict(X_test)
y_true=y_pred

cm=confusion_matrix(y_true,y_pred)
#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Artificial Neural Networks Confusion Matrix")
plt.show()
target_names=['NV', 'MEL', 'BKL', 'DF', 'SCC', 'BCC', 'VASC', 'AK']
print(classification_report(y_test, y_pred, target_names=target_names))
print (algorithms)
print (score)
x_pos = [i for i, _ in enumerate(algorithms)]
plt.bar(x_pos, score, color='green')
plt.xlabel("Algoritmalar")
plt.ylabel("Basari Yuzdeleri")
plt.title("Basari Siralamalar")
plt.xticks(x_pos, algorithms,rotation=90)
plt.show()
ödev_df.groupby('anatom_site_general').size().plot(kind='bar') # alfabetik
plt.show()
ödev_df.groupby('sınıf').size().plot(kind='bar') # alfabetik
plt.show()
ödev_df.groupby('gender').size().plot(kind='bar') # alfabetik
plt.show()
ödev_df.groupby('age_approx').size().plot(kind='bar') # alfabetik
plt.show()
ödev_df.groupby('age_approx').size().plot(kind='bar', color=['firebrick', 'green', 'blue', 'black', 'red',
                    'purple', 'seagreen', 'skyblue', 'black', 'tomato', 'pink', 'orange', 'yellow','gray','turquoise','olive',
                                                             'maroon','violet']) # alfabetik
plt.show()
ödev_df.groupby('gender').size().plot(kind='bar', color=['firebrick', 'green', 'blue', 'black', 'red',
                    'purple', 'seagreen', 'skyblue', 'black', 'tomato', 'pink', 'orange', 'yellow','gray','turquoise','olive',
                                                             'maroon','violet']) # alfabetik
plt.show()
ödev_df.groupby('anatom_site_general').size().plot(kind='bar', color=['firebrick', 'green', 'blue', 'black', 'red',
                    'purple', 'seagreen', 'skyblue', 'black', 'tomato', 'pink', 'orange', 'yellow','gray','turquoise','olive',
                                                             'maroon','violet']) # alfabetik
plt.show()
ödev_df.groupby('sınıf').size().plot(kind='bar', color=['firebrick', 'green', 'blue', 'black', 'red',
                    'purple', 'seagreen', 'skyblue', 'black', 'tomato', 'pink', 'orange', 'yellow','gray','turquoise','olive',
                                                             'maroon','violet']) # alfabetik
plt.show()
grafik=ödev_df[ödev_df['sınıf']==4]
grafik.groupby('gender').size().plot(kind='bar')
plt.show()
grafik=ödev_df[ödev_df['sınıf']==4]
grafik.groupby('anatom_site_general').size().plot(kind='bar')
plt.show()
