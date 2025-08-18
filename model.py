import pandas as pd


datas= pd.read_csv('data/student-mat.csv',sep=';')

#preparing model
#x- inputs/features- dataframe
#y- outputs/targets- series'''
print("\n=== PREPARING FOR ML MODEL ===")
X = datas.iloc[:, 0:32]  # All rows, columns 0-31 (1-32)
Y = datas['G3'] #or y = datas.iloc[:, -1]

#shape of X and Y
'''print("\nX shape:", X.shape)
print("\nY shape:", Y.shape)'''
#head of X and Y
'''print("\nX\n", X.head())
print("\nY\n", Y.head())'''

#datatypes in X
'''
print("Data types in X:")
print(X.dtypes)'''

#Handling non numeric values -
#before mapping 
print("Before mapping:")
print(X['sex'].unique())
#mapping
X['sex']= X['sex'].map({'M':1,'F':0})
#after mapping
print("After mapping:")
print(X['sex'].unique())
print(X['sex'].value_counts())#gives count on each category

#similarly to all binary choices do mapping using dictionaries
#school
X['school']= X['school'].map({'GP':1,'MS':0})
print(X['school'].unique())
print(X['school'].value_counts())

# address
X['address']= X['address'].map({'U':1,'R':0})
print(X['address'].unique())
print(X['address'].value_counts())

# internet
X['internet']= X['internet'].map({'yes':1,'no':0})
print(X['internet'].unique())
print(X['internet'].value_counts())

# romantic
X['romantic']= X['romantic'].map({'yes':1,'no':0})
print(X['romantic'].unique())
print(X['romantic'].value_counts())

# activities
X['activities']= X['activities'].map({'yes':1,'no':0})
print(X['activities'].unique())
print(X['activities'].value_counts())

#famsize
X['famsize']= X['famsize'].map({'GT3':1,'LE3':0})
print(X['famsize'].unique())
print(X['famsize'].value_counts())

#Pstatus
X['Pstatus']= X['Pstatus'].map({'T':1,'A':0})
print(X['Pstatus'].unique())
print(X['Pstatus'].value_counts())

#paid
X['paid']= X['paid'].map({'yes':1,'no':0})
print(X['paid'].unique())
print(X['paid'].value_counts())

#schoolsup
X['schoolsup']= X['schoolsup'].map({'yes':1,'no':0})
print(X['schoolsup'].unique())
print(X['schoolsup'].value_counts())

#famsup
X['famsup']= X['famsup'].map({'yes':1,'no':0})
print(X['famsup'].unique())
print(X['famsup'].value_counts())


#higher
X['higher']= X['higher'].map({'yes':1,'no':0})
print(X['higher'].unique())
print(X['higher'].value_counts())

#Multi-class categorical(more than one options) mapping- one-hot 
        #get_dummies() allows you to represent categorical data in a numerical format(we get multiple colums)

#mothers job
# before encoding 
print("\nbefore encoding shape is")
print(datas['Mjob'].shape) #- (395,)
#encoding
mjob_dummies = pd.get_dummies(datas['Mjob'], prefix='Mjob')
X = X.drop('Mjob', axis=1)  # axis=1 means “remove this column" ,axis = 0 "remove row"
X = pd.concat([X, mjob_dummies], axis=1)  # Add all dummy columns


#fathers job
print("\nbefore encoding shape is")
print(datas['Fjob'].shape) 
#encoding
fjob_dummies = pd.get_dummies(datas['Fjob'], prefix='Fjob')
X = X.drop('Fjob', axis=1)  # axis=1 means “remove this column" ,axis = 0 "remove row"
X = pd.concat([X, fjob_dummies], axis=1)  # Add all dummy columns

# after encoding 
print("\nafter encoding")
print(X.head())


#drop unwanted featues

X = X.drop('reason', axis=1)
X = X.drop('guardian', axis=1)
X = X.drop('nursery', axis=1)

print("Data types in X:")
print(X.dtypes)

'''              '''