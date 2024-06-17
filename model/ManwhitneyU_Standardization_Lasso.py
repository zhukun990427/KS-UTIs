import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split

dataPath = '------------.xlsx'
data = pd.read_excel(dataPath)
data.describe()

y = data['Label']
data_a = data[y == 0]
data_b = data[y == 1]
X = data.iloc[:,2:]


### Manwhitney U-rank sum test ###
colNamesSel_mwU = []
for colName in X_train_a.columns[:]: 
    try:
        if mannwhitneyu(X_train_a[colName],X_train_b[colName])[1] < 0.05:
            colNamesSel_mwU.append(colName)
    except:
        print(colName,'gets error !!')
print(len(colNamesSel_mwU))
print(colNamesSel_mwU)

X_train_mul = X_train[colNamesSel_mwU]
X_test_mul = X_test[colNamesSel_mwU]
X_train_mul


### standardization ###
scaler = StandardScaler()
X_train_mul_scal = scaler.fit_transform(X_train_mul)
X_test_mul_scal = scaler.transform(X_test_mul)
X_train_mul_scal = pd.DataFrame(X_train_mul_scal,columns = colNamesSel_mwU)
X_test_mul_scal = pd.DataFrame(X_test_mul_scal,columns = colNamesSel_mwU)



### LASSO regression ###
alphas = np.logspace(-10, 1, 100, base = 10)
selector_lasso = LassoCV(alphas=alphas, cv = 5, max_iter =5000)
selector_lasso.fit( X_train_mul_scal, y_train)
print(selector_lasso.alpha_)
values = selector_lasso.coef_[selector_lasso.coef_ != 0]
colNames_sel = X_train_mul_scal.columns[selector_lasso.coef_ != 0]


width = 0.45
plt.bar(colNames_sel, values
        , color= 'lightblue'
        , alpha = 1)
plt.xticks(np.arange(len(colNames_sel)),colNames_sel
           , rotation=45 
           , ha = 'right'
          )
plt.ylabel("Coefficient")
plt.show()

MSEs_mean = selector_lasso.mse_path_.mean(axis = 1)
MSEs_std = selector_lasso.mse_path_.std(axis = 1)
plt.figure()
plt.errorbar(selector_lasso.alphas_,MSEs_mean    
             , yerr=MSEs_std                    
             , fmt="o"                          
             , ms=3                             
             , mfc="r"                          
             , mec="r"                          
             , ecolor="lightblue"               
             , elinewidth=2                    
             , capsize=4                       
             , capthick=1)                      
plt.semilogx()
plt.axvline(selector_lasso.alpha_,color = 'black',ls="--")
plt.xlim(1e-3,10)
plt.ylim(-0.1,1.5)
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.show()

coefs = selector_lasso.path(X_train_mul_scal, y_train, alphas=alphas,
                    max_iter = 5000
                           )[1].T
plt.figure()
plt.semilogx(selector_lasso.alphas_,coefs, '-')
plt.axvline(selector_lasso.alpha_,color = 'black',ls="--")
plt.xlim(1e-3,10)
plt.ylim(-0.8,0.8)
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.show()