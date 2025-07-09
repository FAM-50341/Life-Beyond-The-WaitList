# Add prediction probability to test set
X_test_with_proba = X_test.copy()
X_test_with_proba['probability'] = stacking_model.predict_proba(X_test)[:, 1]
X_test_with_proba['Age_Difference'] = X['Age_Difference'].loc[X_test.index]

# Sort for clean line plot
sorted_data = X_test_with_proba.sort_values('Age_Difference')

plt.figure(figsize=(10, 6))
plt.plot(sorted_data['Age_Difference'], sorted_data['probability'], color='blue')
plt.title("Probability of Eligibility vs. Age Difference")
plt.xlabel("Age Difference between Patient and Donor")
plt.ylabel("Predicted Eligibility Probability")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('/content/drive/My Drive/TransplantData/line.png')
