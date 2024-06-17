from sklearn import metrics
import matplotlib.pylab as plt

plt.rc('font', family='Arial')   
plt.figure(dpi=300)

###y_true_* is the true value of the label, and y_score_* is the predicted probability of the label.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

y_true_1 = [-,-,-,-,-,-,-]

y_score_1 = [-,-,-,-,-,-,-]

fpr1, tpr1, thresholds = metrics.roc_curve(y_true_1, y_score_1)
roc_auc1 = metrics.auc(fpr1, tpr1)  # the value of roc_auc1
print(roc_auc1)
plt.plot(fpr1, tpr1, '#00008B', label=' SupportVectorMachine = %0.4f' % roc_auc1)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

y_true_2 = [-,-,-,-,-,-,-]

y_score_2 = [-,-,-,-,-,-,-]

fpr2, tpr2, _ = metrics.roc_curve(y_true_2, y_score_2)
roc_auc2 = metrics.auc(fpr2, tpr2)  # the value of roc_auc1
print(roc_auc2)
plt.plot(fpr2, tpr2, '#006400', label=' Random Forest = %0.4f' % roc_auc2)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

y_true_3 = [-,-,-,-,-,-,-]

y_score_3 = [-,-,-,-,-,-,-]

fpr3, tpr3, _ = metrics.roc_curve(y_true_3, y_score_3)
roc_auc3 = metrics.auc(fpr3, tpr3)  # the value of roc_auc1
print(roc_auc3)
plt.plot(fpr3, tpr3, '#800080', label=' Logistic Regression = %0.4f' % roc_auc3)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'y--')
# plt.xlim([0, 1])  # the range of x-axis
# plt.ylim([0, 1])  # the range of y-axis
plt.xlabel('False Positive Rate')  # the name of x-axis
plt.ylabel('True Positive Rate')  # the name of y-axis
plt.title('Receiver Operating Characteristic ')  # the title of figure
