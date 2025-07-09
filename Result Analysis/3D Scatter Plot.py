from mpl_toolkits.mplot3d import Axes3D

# Use top 3 important features
features_3d = top_features[:3]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color based on prediction
y_pred_test = stacking_model.predict(X_test)
colors = ['red' if y == 0 else 'green' for y in y_pred_test]

ax.scatter(X_test[features_3d[0]],
           X_test[features_3d[1]],
           X_test[features_3d[2]],
           c=colors,
           alpha=0.6,
           marker='o')

ax.set_xlabel(features_3d[0])
ax.set_ylabel(features_3d[1])
ax.set_zlabel(features_3d[2])
ax.set_title("3D Scatter Plot of Top 3 Features (Prediction-Based)")

plt.show()
plt.savefig('/content/drive/My Drive/TransplantData/3D.png')
