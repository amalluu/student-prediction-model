import pandas as pd

#See What We're Working With

datas= pd.read_csv('data/student-mat.csv',sep=';')
print("\n=== BASIC DATA INFO ===")
print("All columns:", datas.columns.tolist())
print("\nDataset shape:", datas.shape)
print("\nBasic statistics:")
print(datas.describe())
print("\nMean of G3:")
print(datas['G3'].mean())

# What We Want to Predict

print("\n=== WHAT ARE WE PREDICTING? ===")
print("Average final grade by studytime:")
print(datas.groupby('studytime')['G3'].mean())

print("Average final grade by absentees:")
print(datas.groupby('absences')['G3'].mean())

print("Average final grade by Mothers education:")
print(datas.groupby('Medu')['G3'].mean())

print("Average final grade by Fathers education:")
print(datas.groupby('Fedu')['G3'].mean())

print("How many students in each study time group:")
print(datas['studytime'].value_counts().sort_index())

print("How many students by parent education:")
print(datas['Medu'].value_counts().sort_index())
print(datas['Fedu'].value_counts().sort_index())