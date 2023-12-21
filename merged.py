import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import random
# Use Hongjin Zhu's N number
random.seed(11993511)
np.random.seed(11993511)

file_url = 'https://raw.githubusercontent.com/HongjinZhu/IDS-Capstone-Project/main/spotify52kData.csv'
df = pd.read_csv(file_url)

################################################## Q1 ##################################################
from scipy.stats import spearmanr
# we do not assume linearity, use spearman correlation
correlation, p_value = spearmanr(df['popularity'], df['duration'])
print(f"Correlation between 'popularity' and 'duration': {correlation}")
print(f"P-value: {p_value}")
sns.scatterplot(x=df['popularity'].values, y=df['duration'].values, size=2, legend=None)
plt.title('scatter plot')
plt.xlabel('popularity')
plt.ylabel('duration')
plt.show()

################################################## Q2 ##################################################
import torch

np.random.seed(11993511)
torch.random.manual_seed(11993511)

# df = pd.read_csv('/content/drive/My Drive/spotify52kData.csv')

# Group by the 'explicit' column and calculate the mean popularity
grouped_data = df.groupby('explicit')['popularity'].mean()
print(grouped_data)

from scipy.stats import ttest_ind
u_statistic, p_value = ttest_ind(
    df[df['explicit'] == True]['popularity'],
    df[df['explicit'] == False]['popularity'],
    alternative='greater'
)
print(f"t test: {u_statistic}, P-Value: {p_value}")

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='explicit', y='popularity', data=df)
plt.title('Popularity of Explicit vs Non-Explicit Songs')
plt.xlabel('Is Explicit')
plt.ylabel('Popularity')
plt.show()

# Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x='explicit', y='popularity', data=df, estimator=mean)
plt.title('Average Popularity of Explicit vs Non-Explicit Songs')
plt.xlabel('Is Explicit')
plt.ylabel('Average Popularity')
plt.show()

################################################## Q3 ##################################################
from scipy.stats import norm

major_keys = df[df['mode'] == 1]['popularity']
minor_keys = df[df['mode'] == 0]['popularity']
plt.figure(figsize=(12, 6))
sns.histplot(major_keys, kde=True, label='Major Keys', color='blue', stat='density')
sns.histplot(minor_keys, kde=True, label='Minor Keys', color='orange', stat='density')
plt.title('Distribution of Popularity for Major and Minor Keys')
plt.xlabel('Popularity')
plt.ylabel('Density')
plt.legend()
plt.show()

from scipy.stats import ttest_ind
# considering heterogeneity of variances, use Welch's t-test
statistic, p_value = ttest_ind(major_keys, minor_keys, alternative='greater', equal_var=False)
print(f"p-value: {p_value}")

################################################## Q4 ##################################################
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Lasso, Ridge, LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve

feature_col = ['duration','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']

y = df['popularity'].values

RMSE = []
betas = []
alphas = []

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
axes = axes.flatten()

for i, f in enumerate(feature_col):
  X = df[f].values.reshape(-1, 1)

  # train test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # hyperparameter tuning
  param_grid = {'alpha': [1e-2, 1e-1, 1, 10, 100, 1000]}
  grid_search = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_squared_error', cv=10)
  grid_search.fit(X_train, y_train)
  best_alpha = grid_search.best_params_['alpha']
  best_model = grid_search.best_estimator_
  beta = best_model.coef_[0]

  # evaluate the model
  pred = best_model.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, pred))

  RMSE.append(rmse)
  betas.append(beta)
  alphas.append(best_alpha)

  axes[i].scatter(X_test, y_test, s=2)
  axes[i].plot(X_test, pred, color='red')
  axes[i].set_title(f'RMSE: {rmse:.2f}, alpha: {best_alpha:.2f}')
  axes[i].set_xlabel(f'{f}')
  axes[i].set_ylabel('popularity')

plt.tight_layout()
plt.show()

# print results
results4 = pd.DataFrame({'feature': feature_col, 'RMSE': RMSE, 'coef': betas, 'best hyper': alphas})
print(results4)

################################################## Q5 ##################################################
X = df[feature_col].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# hyperparameter tuning
param_grid = {'alpha': [1e-2, 1e-1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_squared_error', cv=10)
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']
best_model = grid_search.best_estimator_
beta = best_model.coef_

# evaluate the model
pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))

# print results with regularization
print(f'RMSE: {rmse}, best hyper: {best_alpha}')
results5 = pd.DataFrame({'feature': feature_col, 'coef': betas})
print(results5)

# without regularization
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))

# print results without regularization
print(f'RMSE: {rmse}')
results6 = pd.DataFrame({'feature': feature_col, 'coef': betas})
print(results6)

################################################## Q6 ##################################################
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Select the 10 song features
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Standardizing the features
x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)

# PCA
pca = PCA()
principalComponents = pca.fit_transform(x)

# Determine the number of components
# plot the explained variance to help decide
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum())
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

explained_variances = pca.explained_variance_ratio_
proportion_of_variance = sum(explained_variances[:4])
print("Proportion of variance accounted for by the first 4 principal components:", proportion_of_variance)

# Choose the number of components and re-fit PCA
n_components = 6
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(x)


# Elbow method to determine the number of clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(principalComponents)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.metrics import silhouette_score

range_n_clusters = list(range(2, 11))
silhouette_avg_scores = []

# Kmeans clustering and silhouette scores
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(principalComponents)
    silhouette_avg = silhouette_score(principalComponents, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the silhouette score is: {silhouette_avg:.4f}")

# Plot silhouette
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_avg_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.show()

# optimal number of clusters
optimal_n_clusters = range_n_clusters[silhouette_avg_scores.index(max(silhouette_avg_scores))]
print(f"The optimal number of clusters is: {optimal_n_clusters}")

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(principalComponents)

df['Cluster'] = y_kmeans
df['Genre'] = df['track_genre']

kk = pd.crosstab(df['Cluster'], df['Genre'])
kk.to_csv('kk.csv')

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(principalComponents)

df['Cluster'] = y_kmeans
df['Genre'] = df['track_genre']

pd.crosstab(df['Cluster'], df['Genre'])

################################################## Q7 ##################################################
# assign X and y
X = df['valence'].values.reshape(-1, 1)
y = df['mode'].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# logistic regression
model = LogisticRegression(class_weight='balanced')

# hyperparameter tuning
param_grid = {'C':  [1e-3, 1e-2, 1e-1, 1, 10, 100]}
grid_search_log = GridSearchCV(estimator = model, param_grid = param_grid, scoring="accuracy")
grid_search_log.fit(X_train, y_train)
best_params = grid_search_log.best_params_
print(best_params)

# test using best model
best_model_log = grid_search_log.best_estimator_
y_pred = best_model_log.predict_proba(X_test)[:, 1]

# calculate AUC
auc = roc_auc_score(y_test, y_pred)
# report betas
betas = best_model_log.coef_[0][0]

# separate subplots for each movie
fig, (roc_ax, decision_boundary_ax) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# plot ROC curve
fp, tp, _ = roc_curve(y_test, y_pred)
roc_ax.plot(fp, tp, label=f'AUC = {auc:.3f}, beta = {betas:.2f}')
roc_ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
roc_ax.set_xlabel('Specificity')
roc_ax.set_ylabel('Sensitivity')
roc_ax.set_title(f'ROC curve')
roc_ax.legend()

# plot outcome
decision_boundary_ax.scatter(X, y, s=5, c=y, marker='o')
x_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_values = 1 / (1 + np.exp(-(best_model_log.coef_[0][0] * x_values + best_model_log.intercept_[0])))
decision_boundary_ax.plot(x_values, y_values, color='red', label='sigmoid')

decision_boundary_ax.set_xlabel('valence')
decision_boundary_ax.set_ylabel('mode')
decision_boundary_ax.set_title(f'outcome')
decision_boundary_ax.legend()

plt.show()

param_grid_svm = {'C': [0.001, 0.01, 0.1, 1],
              'kernel': ['linear']}

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# hyperparameter tuning using grid search
grid_search_svm = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)
best_params_svm = grid_search_svm.best_params_
print("Best Hyperparameters:", best_params_svm)

# use model to make predictions and assess accuracy of model
best_svm_pred = grid_search_svm.predict(X_test)

# assess model accuracy by comparing predictions with reality
acc_best_svm = accuracy_score(y_test, best_svm_pred)
print(f'Best SVM model accuracy: {acc_best_svm:.3f}')

# assess model by AUC
svm_model = SVC(kernel='linear', C=0.001, probability=True)
svm_model.fit(X_train, y_train)
best_svm_probs = svm_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, best_svm_probs)
print(f'Best SVM model AUC: {auc:.4f}')

# plot ROC curve
fpr, tpr, _ = roc_curve(y_test, best_svm_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
plt.title('ROC Curve')
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

from scipy.stats import chi2_contingency

mean_valence = df['valence'].mean()
df['valence_group'] = df['valence'].apply(lambda x: 1 if x >= mean_valence else 0)

# contingency table
contingency_table = pd.crosstab(df['valence_group'], df['mode'])

# chi-square test
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-square value: {chi2}")
print(f"p-value: {p}")
print(contingency_table)

"""We can predict whether a song is in major or minor key from valence using logistic regression or a support vector machine, but the results are not good.
After performing a Chi-square test we find that there is no significant difference between the mode of the songs with high and low valence.
"""

# try decision tree
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

y_pred = tree.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f'Best decision tree AUC: {auc:.3f}')

# plot ROC curve
fpr, tpr, _ = roc_curve(y_test, best_svm_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
plt.title('ROC Curve')
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

################################################## Q8 ##################################################
##choose the pca features
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#X = principalComponents[:, :4]
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[features].values

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into features and target
y = df['track_genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# convert to PyTorch tensors
y_train = torch.tensor(y_train_encoded, dtype=torch.long)
y_test = torch.tensor(y_test_encoded, dtype=torch.long)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

from torch.utils.data import TensorDataset, DataLoader

# data loaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

num_features = 10
num_classes = len(torch.unique(y_train))

model = Net(num_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
ls = []
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    ls.append(loss.item())

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test set: {accuracy}%')
# Plot the loss over time
import matplotlib.pyplot as plt
plt.plot(ls)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# predict probabilities for each class
outputs = model(X_test)
probabilities = torch.softmax(outputs, dim=1).detach().numpy()

num_classes = len(np.unique(y_test.numpy()))
y_test_binarized = label_binarize(y_test.numpy(), classes=range(num_classes))

auc_scores = []
for i in range(num_classes):
    class_auc = roc_auc_score(y_test_binarized[:, i], probabilities[:, i])
    auc_scores.append(class_auc)

average_auc = np.mean(auc_scores)
print(f"Average AUC: {average_auc}")

################################################## Q9 ##################################################
file_path2 = 'starRatings.csv'
cols = [i for i in range(5000)]
ratings_df = pd.read_csv(file_path2, names=cols)

songs_df = df.head(5000)

mean_ratings = ratings_df.mean(axis=0)

popularity = songs_df['popularity']

from scipy.stats import spearmanr

# test correlation between average ratings and popularity
# data non-linear, use spearman correlation
corr, p_value = spearmanr(mean_ratings, popularity)
print(f"Spearman correlation coefficient: {corr}")
print(f"p-value: {p_value}")

# plot the correlation
sns.scatterplot(x=popularity, y=mean_ratings, size=1, alpha=0.7, palette='viridis', legend=None)
sns.regplot(x=popularity, y=mean_ratings, scatter=False, color='gray')
plt.title('popularity vs. average rating')
plt.xlabel('popularity')
plt.ylabel('average rating')
plt.show()

song_popularity = pd.DataFrame({'song': popularity.index, 'average_rating': mean_ratings.values, 'popularity': popularity.values})
sorted_songs = song_popularity.sort_values(by='average_rating', ascending=False)
rec_songs = sorted_songs.head(10)
print(rec_songs)

################################################## Q10 ##################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Train-test split
ratings_df = ratings_df.fillna(0)
train_users, test_users = train_test_split(ratings_df.index, test_size=0.2, random_state=42)
train_df = ratings_df.loc[train_users]
test_df = ratings_df.loc[test_users]

# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(train_df)
np.fill_diagonal(similarity_matrix, 0)

similarity_matrix_train = cosine_similarity(train_df)
np.fill_diagonal(similarity_matrix_train, 0)

# use cosine similarity
def predict_rating(user_index, item_index, train_ratings, similarity):
    sim_scores = similarity[user_index]
    item_ratings = train_ratings[:, item_index]
    valid_scores = sim_scores[item_ratings > 0]
    valid_ratings = item_ratings[item_ratings > 0]
    if valid_scores.sum() > 0:
        return np.dot(valid_scores, valid_ratings) / valid_scores.sum()
    else:
        return 0

predictions = []
for user_index in range(test_df.shape[0]):
    user_predictions = []
    for item_index in range(test_df.shape[1]):
        predicted_rating = predict_rating(user_index, item_index, train_df.values, similarity_matrix_train)
        user_predictions.append(predicted_rating)
    predictions.append(user_predictions)

predictions_df = pd.DataFrame(predictions, index=test_df.index, columns=test_df.columns)

def get_top_10_ratings_per_user(df):
    top_10_per_user = []
    for i in df.index:
        top_items = df.loc[i].nlargest(10).index.tolist()
        top_10_per_user.append(top_items)

    return np.array(top_10_per_user)


top_10_ratings_test = get_top_10_ratings_per_user(test_df)
top_10_ratings_predictions = get_top_10_ratings_per_user(predictions_df)
print(test_df.shape)

# lab code
def average_precision_at_k(actual, predicted, k=10):
  k = min(k,min(len(actual), len(predicted))) # at max 100, because we are predicting only top 100 movies for each user
  if len(actual) > k: # e.g. take the top 10 movies for each user based on actual data
    actual = actual[:k]
  if len(predicted) > k: # e.g. subset top 10 movies for that user
    predicted = predicted[:k]

  average_precision = 0.0

  # List of T/F values corresponding to every predicted movie indicating whether it is in the ground truth or not.
  relevant_inds = np.asarray([movie in actual for movie in predicted]) # for every movie index in the actuals, is the movie in predicted
  print ("relevant_inds", relevant_inds)

  num_relevant = np.sum(relevant_inds*1) # k-sized array [T, F ...  T] or [1, 0 ... 1]
  print ("num_relevant", num_relevant)

  if num_relevant == 0: # if no trues, then return 0
    return 0

  #  precision calculated at each index i of the predicted list.
  # Precision at index i is the number of relevant items found in the top i+1 predictions divided by i+1
  precision_at_k = np.asarray([(np.sum(relevant_inds[:i+1]*1.0)/(i+1)) for i in range(k)])
  print ("precision_at_k", precision_at_k)

  for ind in range(k):
    if relevant_inds[ind]:
      average_precision += precision_at_k[ind]

  average_precision /= num_relevant
  return average_precision
def mean_average_precision_at_k(actual, predicted, k=10):
  # Here, actual/ground truth will be of the shape (num_users, num_movies)
  # We will run the average_precision_at_k function for every user and then take the mean of those values

  mean_average_precision = 0.0

  for user_ind in range(len(actual)):
    user_recommendation = predicted[user_ind]
    user_actual = actual[user_ind]

    ap = average_precision_at_k(user_actual, user_recommendation, k)

    mean_average_precision += ap

  mean_average_precision /= len(actual)

  return mean_average_precision

input_array = np.array([3877, 3003, 2260, 2562, 3216, 2105, 2003, 2011, 3464, 3253])

result_array = np.tile(input_array, (2000, 1))

print(result_array.shape)

def _average_precision_at_k(actual, predicted, k=10):
    if len(predicted) < k:
        k = len(predicted)

    precision_at_i = []
    relevant_count = 0
    
    for i in range(k):
        if predicted[i] in actual:
            relevant_count += 1
            precision_at_i.append(relevant_count / (i + 1))
    
    if not precision_at_i:
        return 0.0
    
    return sum(precision_at_i) / min(k, len(actual))

avg_precision = _average_precision_at_k(top_10_ratings_test, result_array)
print(avg_precision)