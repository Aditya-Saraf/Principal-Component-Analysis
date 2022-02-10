

#Allowed lib - Numpy, Pandas, Mat-plotlib, SciPy, and Sklearn
# scikit-learn = 0.24.2
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as npla
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# UTILITY FUNCTIONS

# In[129]:


#==================================================================|
#------------------UTILITY FUNCTIONS START-------------------------|
#==================================================================|

def normalise_df(X_train,X_test):
    train_max_vector = X_train.max()
    train_min_vector = X_train.min()
    train_range = train_max_vector - train_min_vector
    normalised_train = X_train - train_min_vector
    normalised_train = normalised_train / train_range
    normalised_test = X_test - train_min_vector
    normalised_test = normalised_test / train_range
    return normalised_train, normalised_test

def dim_reduction(p,df,cov_mat):
    e_val_frame, e_vec_frame = get_eigen_val_vec(cov_mat)
    e_val = np.array(e_val_frame)
    e_vec = np.array(e_vec_frame)
    e_val = e_val[:p]
    e_vec = e_vec[:,:p]
    reduced_data_matrix = np.matmul(np.array(df),e_vec)
    return reduced_data_matrix

def display_df_5x5(df):
    df_rx5 = df[df.columns[:5]]
    df_5x5 = df_rx5.head(5)
    print(df_5x5)

def covariance_mat(df):
    cov_mat = df.cov()
    return cov_mat

def get_eigen_val_vec(df):
    np_ar = np.array(df)
    eigenValues, eigenVectors = npla.eig(np_ar)
    idx = np.argsort(eigenValues)[::-1]  
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    e_val_frame = pd.Series(eigenValues)      
    e_vec_frame = pd.DataFrame(eigenVectors)
    return e_val_frame, e_vec_frame
  
def plot_eigen_values(e_val,number_of_PC_to_show):
    fig = plt.figure(figsize=(16,15))
    sing_vals = np.arange(number_of_PC_to_show) + 1
    plt.plot(sing_vals, (np.array(e_val))[:number_of_PC_to_show], 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    
    leg = plt.legend(['Eigenvalues vs Principal component'], loc='best', borderpad=0.3, 
                  shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                  markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    
    for i, j in zip(sing_vals[:10], e_val[:10]):
        plt.text(i + 0.01, j + 0.01, '({}, {})'.format(i, round(j,2)))
    plt.show()
    

def plot_variance_covered(e_val,number_of_PC_to_show):
    s = e_val.sum()
    e_ar = np.array(e_val)
    e_ar = e_ar[:number_of_PC_to_show]
    var_cov = 0
    var = []
    for e in e_ar:
        var_cov = var_cov + (e*100)/s
        var.append(var_cov)
    var_ar = np.array(var)
    fig = plt.figure(figsize=(16,10))
    sing_vals = np.arange(number_of_PC_to_show) + 1
    plt.plot(sing_vals, (np.array(var_ar))[:number_of_PC_to_show], 'ro-', linewidth=2)
    plt.title('Number of components needed to eplain Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance (%)')
    leg = plt.legend(['Eigenvalues vs Principal component'], loc='best', borderpad=0.3, 
                  shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                  markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    
    for i, j in zip(sing_vals[:15], var_ar[:15]):
        plt.text(i + 0.03, j + 0.03, '({}, {})'.format(i, round(j,2)))
  

    plt.show()

def predict_KNN(k,X_train,X_test,Y_train,Y_test):
    knn = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')
    knn.fit(X_train,Y_train)
    Y_pred = knn.predict(X_test)
    return Y_pred

def get_accuracy(Y_pred, Y_test):
    score = metrics.accuracy_score(Y_test,Y_pred)
    return score

def get_knn_accuracy(k,X_train,X_test,Y_train,Y_test):
    Y_pred = predict_KNN(k,X_train,X_test,Y_train,Y_test)
    score = get_accuracy(Y_pred, Y_test)
    return score

def get_accuracy_list(total_k_vals,X_train,X_test,Y_train,Y_test):
    score_list = []
    for k in range(1,total_k_vals+1):
        score = get_knn_accuracy(k,X_train,X_test,Y_train,Y_test)
        score_list.append(score)
        score_ar = np.array(score_list)
    return score_ar

def plot_k_vs_accuracy(accuracy_ar):
    max_index_row = np.argmax(accuracy_ar, axis=0)
    fig = plt.figure(figsize=(16,10))
    nk = len(accuracy_ar)
    sing_vals = np.arange(nk) 
    plt.plot(sing_vals, (np.array(accuracy_ar))[:nk], 'ro-', linewidth=2)
    plt.title('Best accuracy at k = ' + str(max_index_row )+ 'N neighbours with Accuracy = ' + str(accuracy_ar[max_index_row]))
    plt.xlabel('K-value')
    plt.ylabel('Accuracy')
    leg = plt.legend(['Accuracy vs K-value in KNN'], loc='best', borderpad=0.3, 
                  shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                  markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    
    for i, j in zip(sing_vals[:10], accuracy_ar[:10]):
        plt.text(i + 0.01, j + 0.01, '({}, {})'.format(i, round(j,2)))
   
    plt.show()

def create_p_PC(p,X_test,Y_test,Y_pred,name_of_csv_file):
    ev_ar = np.array(X_test)
    ev_ar = ev_ar[:,:p]
    pc_mat =  pd.DataFrame(ev_ar, columns = ['PC'+str(x) for x in range(0,ev_ar.shape[1])])
    pc_mat['Y_original'] = Y_test
    pc_mat['Y_predicted'] = Y_pred
    pc_mat.to_csv(index=False)
    compression_opts = dict(method='zip',
                            archive_name=name_of_csv_file+'.csv')  
    pc_mat.to_csv(name_of_csv_file+'.zip', index=False,
              compression=compression_opts) 
    print(pc_mat)

def get_acc_list_wrt_p(k,p_list, normalised_train, normalised_test, cov_mat, Y_train, Y_test):
    acc_list = []
    for p in p_list:
        PC_to_consider = p
        reduced_X_train = dim_reduction(PC_to_consider,normalised_train,cov_mat)
        reduced_X_test = dim_reduction(PC_to_consider,normalised_test,cov_mat)
        acc_value = get_knn_accuracy(k,reduced_X_train,reduced_X_test,Y_train,Y_test)
        acc_list.append(acc_value)
    return acc_list

def plot_p_vs_acc(k,acc_list,p_list):
    max_index_row = np.argmax(acc_list, axis=0)
    fig = plt.figure(figsize=(16,10))
    nk = len(acc_list)
    sing_vals = p_list 
    plt.plot(sing_vals, (np.array(acc_list))[:nk], 'ro-', linewidth=2)
    plt.title('Best accuracy at p = ' + str(p_list[max_index_row]) + ' for '+ str(k) + ' NN with Accuracy = ' +str(acc_list[max_index_row]))
    plt.xlabel('No. of PC used for PCA (p)')
    plt.ylabel('Accuracy for ' +str(k)+ ' NN')
    leg = plt.legend(['Accuracy vs p'], loc='best', borderpad=0.3, 
                  shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                  markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    
    for i, j in zip(sing_vals[:10], (np.array(acc_list))[:10]):
        plt.text(i + 0.01, j + 0.001, '({}, {})'.format(i, round(j,2)))

    plt.show()

def standardise_df(X_train,X_test):
    train_mean = X_train.mean()
    train_sd = X_train.std()
    standardised_train = X_train - train_mean
    standardised_train = standardised_train / train_sd
    standardised_test = X_test - train_mean
    standardised_test = standardised_test / train_sd
    return standardised_train, standardised_test

def display_variance_covered(e_val,n_e):
    s = e_val.sum()
    e_ar = np.array(e_val)
    e_ar = e_ar[:n_e]
    var = []
    for e in e_ar:
        var_cov = (e*100)/s
        var.append(var_cov)
    var_ar = np.array(var)
    d = {'Eigen Values': e_ar, 'Variance Covered': var_ar}
    df = pd.DataFrame(data = d)
    print(df)

#==================================================================|
#------------------UTILITY FUNCTIONS END---------------------------|
#==================================================================|


# LOAD DATA

# In[130]:


#LOAD DATA
train_data_path ="pca_train.csv"
test_data_path ="pca_test.csv"

df_train = pd.read_csv(train_data_path)
df_test = pd.read_csv(test_data_path)
X_train = df_train.copy()
X_test = df_test.copy()
del X_train['Class']
del X_test['Class'] 
Y_train = df_train['Class']
Y_test = df_test['Class']


# DATA UNDERSTANDING

# In[131]:


#(a) Number of data points with their respective classes
test_rows_c0 = df_test[df_test["Class"] == 0]
test_rows_c1 = df_test[df_test["Class"] == 1]
train_rows_c0 = df_train[df_train["Class"] == 0]
train_rows_c1 = df_train[df_train["Class"] == 1]


print(">>> Size of Training set = ",df_train.shape)
print(">>> Size of Testing set = ",df_test.shape)
print(">>> Total no. of test data points with class-0 = ",test_rows_c0.shape[0])
print(">>> Total no. of test data points with class-1 = ",test_rows_c1.shape[0])
print(">>> Total no. of train data points with class-0 = ",train_rows_c0.shape[0])
print(">>> Total no. of train data points with class-1 = ",train_rows_c1.shape[0])


# DATA PREPROCESSING > NORMALISATION

# In[132]:


#(b)--------------------NORMALISED DATA-------------------------------|
#min/max Normalisation on training and testing data
normalised_train, normalised_test = normalise_df(X_train,X_test)


# PCA ON NORMALISED DATA

# In[133]:


#(i)Covariance matrix
cov_mat = covariance_mat(normalised_train)
print("\n>>> Size of Resultant Covariance Matrix = ", cov_mat.shape)
print("\n>>> First 5x5 of covariance matrix = ")
display_df_5x5(cov_mat)


# In[134]:


#(ii)Eigen Values and Eigen Vectors
e_val, e_vec = get_eigen_val_vec(cov_mat)
print("\n>>> Largest 5 eigen values = ")
print(e_val.head(5))


# In[135]:


#(iii)plot_eigen_values(eigen_values,total PC to display)
plot_eigen_values(e_val,30)
plot_variance_covered(e_val,15)
#print('Observation: Looking at the scree plot and the plot of variance covered,\nwe can assume that 9 eigen vectors are sufficient. As we can see from\nthe variance covered by largest 15 eigen values below-')
display_variance_covered(e_val,15)
#print('PCs associated with the eigen vectors from largest 9 eigen\nvalues shall cover almost 95% of the total variance.')


# Observation: 
#     
# >Looking at the scree plot and the plot of variance covered,we can assume that 9 eigen vectors are 
# sufficient. 
# 
# >As we can see from the table of variance covered by largest 15 eigen values above, PCs associated with the eigen vectors from largest 9 eigen values shall cover almost 95% of the total variance.
# 

# KNN ON DIMENSIONALLY REDUCED DATA

# In[136]:


#KNN for k=5 and PCA with p=10
PC_to_consider = 10
k = 5
reduced_X_train = dim_reduction(PC_to_consider,normalised_train,cov_mat)
reduced_X_test = dim_reduction(PC_to_consider,normalised_test,cov_mat)
Y_pred = predict_KNN(k,reduced_X_train,reduced_X_test,Y_train,Y_test)
acc_value = get_accuracy(Y_test,Y_pred)
print(">>> Accuracy when k=5 and p=10 is: " + str(acc_value) + ' (on the scale of zero to one)\n')


# Observation:   
# >Accuracy of KNN (k=5) on given dataset with 10 Principal Components comes out to be 98% (approx). 

# In[137]:


#Create .zip file in current directory. Extract the zip file for .csv
create_p_PC(PC_to_consider,reduced_X_test,Y_test,Y_pred,"PC_Normalised_data")


# OPTIMISATION

# In[138]:


#Varying p values
p_list = [2, 4, 8, 10, 20, 25, 30]
k=5 
acc_list = get_acc_list_wrt_p(k,p_list, normalised_train, normalised_test, cov_mat, Y_train, Y_test)
plot_p_vs_acc(k,acc_list,p_list)


# Observation:
# >Based on the p vs accuracy graph, we can say that using 10 PCs gives best accuracy. Hence 10 is the
# most reasonable number of pricipal components that should be employed.
# 
# >However, we might get similar acuracy when p = 25 or 30. But the whole point of using PCA is to reduce 
# the dimensions while maintaining the accuracy which is perfectly achieved while using 10 PCs.

# DATA PREPROCESSING > STANDARDISATION

# In[139]:


#(c)--------------------STANDARDISED DATA-------------------------------|
#Standardisation
standardised_train, standardised_test = standardise_df(X_train,X_test)


# PCA ON STANDARDISED DATA

# In[140]:


#Covariance matrix
cov_mat = covariance_mat(standardised_train)
print("\n>>> Size of Resultant Covariance Matrix = ", cov_mat.shape)
print("\n>>> First 5x5 of covariance matrix = ")
display_df_5x5(cov_mat)


# In[141]:


#Eigen Values and Eigen Vectors
e_val, e_vec = get_eigen_val_vec(cov_mat)
print("\n>>> Largest 5 eigen values = ")
print(e_val.head(5))


# In[142]:


#plot_eigen_values(eigen_values,total PC to display)
plot_eigen_values(e_val,30)
plot_variance_covered(e_val,15)
display_variance_covered(e_val,15)


# Observation: 
# >Looking at the scree plot and the plot of variance covered, we can assume that 11 eigen vectors are sufficient to create a KNN predictive model based on Principal Components. 
# 
# >As we can see from the variance covered by largest 15 eigen values above, PCs associated with the eigen vectors from largest 11 eigen values shall cover more than 96% of the total variance.
# 

# KNN ON DIMENSIONALLY REDUCED DATA

# In[143]:


#KNN for k=5 and p=10
PC_to_consider = 10
k = 5
reduced_X_train = dim_reduction(PC_to_consider,standardised_train,cov_mat)
reduced_X_test = dim_reduction(PC_to_consider,standardised_test,cov_mat)
Y_pred = predict_KNN(k,reduced_X_train,reduced_X_test,Y_train,Y_test)
acc_value = get_accuracy(Y_test,Y_pred)
print(">>> Accuracy when k=5 and p=10 is: " + str(acc_value) + ' (on the scale of zero to one)\n')


# Observation:   
# >Accuracy of KNN (k=5) on given dataset with 10 Principal Components comes out to be 94% (approx). 

# In[144]:


#Create .zip file in current directory. Extract the zip file for .csv
create_p_PC(PC_to_consider,reduced_X_test,Y_test,Y_pred,"PC_Standardised_data")


# OPTIMISATION

# In[146]:


#Varying p values
p_list = [2, 4, 8, 10, 20, 25, 30]
k=5 
acc_list = get_acc_list_wrt_p(k,p_list, standardised_train, standardised_test, cov_mat, Y_train, Y_test)
plot_p_vs_acc(k,acc_list,p_list)


# Observation:    
# >Based on the p vs accuracy graph, we can say that using 2 PCs gives best accuracy. But if noticed, the variance covered by 2 PCs (~64%) is too low, which might induce a bias. Hence we must choose more than 2 PCs.
# 
# >After 2 PCs, we have the choice between p=10 or p=20, which are sufficient enough to cover variance of more than 95%. But taking p=10 compromises with the accuracy with accuracy = 94%. However taking p=20 gives us 97% of accuracy but on the same side consuming more computation power and lowering the interpretability of data.
# 
# >Hence, it is a tradeoff between computational power and data interpretability versus accuracy where p=20 consumes more computational power & less interpretable but gives better accuracy, whereas p=10 shall take significantly less computational power and will be significantly more interpretable than p=20, but all at the cost of accuracy.
# 
# >After considering all these factors, p=10 seems to be the most reasonalbe number of PCs that should be used to model KNN in this case.
# 
# 
