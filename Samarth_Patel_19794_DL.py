import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../data/04_cricket_1999to2011.csv')   # Input Path
df2 = df[['Match', 'Date', 'Innings', 'Over', 'Total.Runs','Innings.Total.Runs','Wickets.in.Hand','Runs.Remaining','Runs']]
df2 = df2[df2['Innings'] == 1] # Only considering the First Inning Score

# Cleaning the Total.Runs column by replacing it with cumulative sum of Runs Column for given match
Match = pd.unique(df2['Match'])
df_clean = df2.groupby('Match') 
last = pd.DataFrame() # Initializing empty data frame to store to clean data  
for i in Match:
    new = df_clean.get_group(i)
    sum_per_over = new['Runs'].cumsum()
    new['Total.Runs'] = sum_per_over
    last = pd.concat([last,new])
for i in range(len(last)):
    last['Runs.Remaining'] = last['Innings.Total.Runs'] - last['Total.Runs']  # Changing Runs.Remaining Column after cleaning Total.Runs Column

# Adding the row which contains situation of match when zero overs were used    
def add_row(x):
    last_row = x.iloc[0]
    last_row['Total.Runs'] = 0
    last_row['Over'] = 0
    last_row['Runs.Remaining'] = last_row['Innings.Total.Runs']  
    last_row['Wickets.in.Hand'] = 10
    return x.append(last_row)
df3 =  last.groupby('Match').apply(add_row).reset_index(drop=True)
df3.reset_index(inplace=True,drop = True)

print('Pre Processing Completed')
w = [] # List for storing wickets
for i in range(1,11):
    w.append(i)
df4 = df3.groupby(by='Wickets.in.Hand')

# Initilization is done by considering the max score for a given wicket in a given match and median of that values are taken to avoid any outlier effect
print('Initializing Parameters')
def Initilization_for_given_wicket(data_frame, wickets):
    data_frame = df4.get_group(i)
    initial_guess = np.median(data_frame.groupby(['Match'])['Runs.Remaining'].max())
    return initial_guess

DL_parameters = [] # Initializing empty List to store the parameters
for i in w:
        DL_parameters.append(Initilization_for_given_wicket(df4, i)) # Initial Guess for a given wicket
DL_parameters.append(9) # Initial Guess for L

print('Optimizing')

# DuckworthLewis Model to get the Average runs obtainable
def DuckworthLewis(Z0, L, u):
    Average_runs_obtainable = Z0 * (1 - np.exp(-L*u/Z0))
    return Average_runs_obtainable

def mse_error(DL_parameters):
    
    Z0 = DL_parameters[:10]
    L = DL_parameters[-1]
    mse = 0
    
    for j in w:
        df_new = df4.get_group(j).reset_index()
        row = len(df_new)
        
        for i in range(row):
            z = DuckworthLewis(Z0[j-1], L, 50-df_new['Over'][i])
            mse += (z - df_new['Runs.Remaining'][i])**2
        
    mse /= len(df3)
    #print(mse)
    return mse

optimize = optimize.minimize(mse_error, DL_parameters, method= 'L-BFGS-B')
#print('Z0')
Z0 = optimize.x[:len(optimize.x)-1]
#print(optimize.x[:len(optimize.x)-1])

L = optimize.x[-1]
print('L = ' + str(optimize.x[-1]))
print('mse = ' + str(optimize.fun))
Dict = {'Wicket_remaining':['Wicket_remaining-1','Wicket_remaining-2','Wicket_remaining-3','Wicket_remaining-4','Wicket_remaining-5','Wicket_remaining-6','Wicket_remaining-7','Wicket_remaining-8','Wicket_remaining-9','Wicket_remaining-10']
       ,'Z0':[Z0[0],Z0[1],Z0[2],Z0[3],Z0[4],Z0[5],Z0[6],Z0[7],Z0[8],Z0[9]]}
Result = pd.DataFrame(Dict)
print(Result) # Results DataFrame

# Plotting 
plt.figure()
u = np.linspace(0,50,500)
for i in Z0:
    y = []
    for j in u:
        y.append(i * (1 - np.exp(-L*(50-j)/i)))
    plt.plot(u,y,label=str(i))
plt.grid()
plt.xlim((0, 50))
plt.ylim((0, 250))
plt.xticks([5*i for i in range(11)])
plt.yticks([10*i for i in range(26)]) 
plt.xlabel('Overs Used')
plt.ylabel('Average runs obtainable')
plt.legend([str(i+1) for i in range(10)])



plt.figure()
u = np.linspace(0,50,500)
for i in Z0:
    y = []
    for j in u:
        y.append(i * (1 - np.exp(-L*(j)/i)))
    plt.plot(u,y,label=str(i))
plt.grid()
plt.xlim((0, 50))
plt.ylim((0, 250))
plt.xticks([5*i for i in range(11)])
plt.yticks([10*i for i in range(26)]) 
plt.xlabel('Overs Remaining')
plt.ylabel('Average runs obtainable')
plt.legend([str(i+1) for i in range(10)])
plt.show()