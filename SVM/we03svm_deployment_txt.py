import pickle
import pandas as pd

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the progrintam and print the current working directory.
# import os
# exit(os.getcwd())

riding_mower = pickle.load(open(r'C:/Users/hiran/DSP 6251/best_svm_model.pkl',"rb"))

print("\n*****************************************************")
print("* The riding mower prediction model *")
print("*****************************************************\n")
Income= float(input("Enter the income"))
Lot_Size=float(input("Enter the lot size of individual"))
data_df = pd.DataFrame({'Income': [Income],'Lot_Size':[Lot_Size]})
outcome = riding_mower.predict(data_df)
probability = riding_mower.predict_proba(data_df)
ownership = ('Nonowner', 'Owner')
print(f"\nThe riding mower model indicates probability of predictions at the {probability[0][1]:.4f}, gives that he/she is {ownership[outcome[0]]}.\n")