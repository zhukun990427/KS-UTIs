import numpy as np
import pandas as pd  
from matplotlib import pyplot as plt
import scipy.stats as st
from sklearn import metrics
from pathlib import Path  
 
class DelongTest():
    def __init__(self,preds1,preds2,label,threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1=preds1
        self._preds2=preds2
        self._label=label
        self.threshold=threshold
        self._show_result()
 
    def _auc(self,X, Y)->float:
        return 1/(len(X)*len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])
 
    def _kernel(self,X, Y)->float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y==X else int(Y < X)
 
    def _structural_components(self,X, Y)->list:
        V10 = [1/len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01
 
    def _get_S_entry(self,V_A, V_B, auc_A, auc_B)->float:
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
    
    def _z_score(self,var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB )**(.5)+ 1e-8)
 
    def _group_preds_by_label(self,preds, actual)->list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y
 
    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)
 
        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)
 
        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)
 
        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)+ self._get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))
 
        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z))*2
 
        return z,p

    def get_z_p(self):   
        return self._compute_z_p()  
 
    def _show_result(self):
        z,p=self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.5f};")
        if p < self.threshold :print("There is a significant difference")
        else:        print("There is NO significant difference")
 
 
    
###true_labels is the true value of the tag 
###model_probs is the predicted probability of the tag.
true_labels = [-,-,-,-,-,-,-]  
model_probs = [{'model_name': 'Radiomics Features', 'probs': [-,-,-,-,-,-,-]} 
              ,{'model_name': 'Clinical Features', 'probs': [-,-,-,-,-,-,-]} 
              ,{'model_name': 'Combined Features', 'probs': [-,-,-,-,-,-,-]} 
               ] 

delong_results = {}  
# Initializes a DataFrame to store the result
df = pd.DataFrame(columns=['Model Pair', 'z_score', 'p_value', 'Significant Difference', 'File Path'])  
  
for i in range(len(model_probs)):  
    for j in range(i+1, len(model_probs)):  
        model1_name = model_probs[i]['model_name']  
        model2_name = model_probs[j]['model_name']  
        model1_probs = model_probs[i]['probs']  
        model2_probs = model_probs[j]['probs']  
  
        delong_test_result = DelongTest(model1_probs, model2_probs, true_labels)  
  
        z, p = delong_test_result.get_z_p()  
  
        delong_results[f'{model1_name} vs {model2_name}'] = {'z_score': z, 'p_value': p}  
  
        significant_difference = p < 0.05  
        file_path = ''  
        df = df.append({'Model Pair': f'{model1_name} vs {model2_name}',  
                        'z_score': z,  
                        'p_value': p,  
                        'Significant Difference': significant_difference,  
                        'File Path': file_path},  
                       ignore_index=True)  
  
# Print result
for key, value in delong_results.items():  
    print(f"{key}: z_score={value['z_score']:.5f}, p_value={value['p_value']:.5f}")  
  
# Save the DataFrame to an Excel file
excel_file_path = 'D:\\-----\\-----.xlsx'  
df.to_excel(excel_file_path, index=False, engine='openpyxl')  
  
print(f"DeLong test results saved to：{excel_file_path}")
