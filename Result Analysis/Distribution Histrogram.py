# Get predicted probabilities
proba = stacking_model.predict_proba(X_test)[:, 1]
labels = y_test.reset_index(drop=True)

plt.figure(figsize=(10, 6))
sns.histplot(proba[labels == 1], bins=20, kde=True, color='green', label='Eligible', stat='density')
sns.histplot(proba[labels == 0], bins=20, kde=True, color='red', label='Not Eligible', stat='density')
plt.title("Distribution of Predicted Probabilities by Class")
plt.xlabel("Predicted Eligibility Probability")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('/content/drive/My Drive/TransplantData/histogram.png')
