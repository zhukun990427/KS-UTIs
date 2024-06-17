import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib as mpl

# Define the true labels and predicted scores for each model
###y_true is the true value of the tag 
###scores_model* is the predicted probability of the tag.
y_true = [-,-,-,-,-,-,-]
scores_model1 = [-,-,-,-,-,-,-] 
scores_model2 = [-,-,-,-,-,-,-]
scores_model3 = [-,-,-,-,-,-,-]


# Define the threshold range for calculating net benefit
thresh_group = np.arange(0, 1, 0.01)

# Define the functions for calculating net benefit
def calculate_net_benefit_model(thresh_group, y_pred_label, y_true):
    net_benefit_model = np.array([])
    y_pred_label = np.array(y_pred_label)  # Convert to NumPy array
    for thresh in thresh_group:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label > thresh).ravel()
        n = len(y_true)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def calculate_net_benefit_all(thresh_group, y_true):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_true, y_true).ravel()
    total = len(y_true)
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, model_name, color='black', first_model=False):
    # Plot the model curve
    ax.plot(thresh_group, net_benefit_model, label=model_name, color=color)

    # Add the "Treat all" curve if this is the first model
    if first_model:
        ax.plot(thresh_group, net_benefit_all, color='black', label='Treat all', linestyle='-')
        ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')
    else:
        ax.plot(thresh_group, net_benefit_all, color='black', linestyle=':')

    # Figure Configuration
    ax.set_xlim(0, 1)
   # ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)
    ax.set_ylim(-0.5,0.5)
    ax.set_xlabel('Threshold Probability', fontdict={'family': 'Arial', 'fontsize': 14})
    ax.set_ylabel('Net Benefit', fontdict={'family': 'Arial', 'fontsize': 14})
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))

    return ax

# Calculate the net benefit for each model
net_benefit_model1 = calculate_net_benefit_model(thresh_group, scores_model1, y_true)
net_benefit_model2 = calculate_net_benefit_model(thresh_group, scores_model2, y_true)
net_benefit_model3 = calculate_net_benefit_model(thresh_group, scores_model3, y_true)
#net_benefit_model4 = calculate_net_benefit_model(thresh_group, scores_model4, y_true)


# Calculate the net benefit for treating all patients
net_benefit_all_1 = calculate_net_benefit_all(thresh_group, y_true)

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.style'] = 'normal'  

# Plot the DCA curve for each model
plt.figure(dpi=1200)
fig, ax = plt.subplots(figsize=(8, 6))

plot_DCA(ax, thresh_group, net_benefit_model1, net_benefit_all_1, 'Radiomics Features', color='#FF1493')
plot_DCA(ax, thresh_group, net_benefit_model2, net_benefit_all_1, 'Clinical Features', color='#A9A9A9')
plot_DCA(ax, thresh_group, net_benefit_model3, net_benefit_all_1, 'Combined Features',  color='#A52A2A',first_model=True)


# Add the legend
ax.legend()

# Show the plot
plt.show()
