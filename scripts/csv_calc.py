import pandas as pd 
import numpy as np
import os

final=[]
lis_of_vid = [csv for csv in os.listdir('path to csv folder') if img.endswith(".csv")]

for path in lis_of_vid:
  if path[-4:] != '.csv':
      continue
  data = pd.read_csv('path to csv folder'+path) 
  data.head()
  result = np.array([0,0,0,0,0])
  for i in range(0,500,5):
    result=result + (data.iloc[i:i+5,-1]).to_numpy()
  result=result/100
  my_result={'AUC_Judd':result[0],'NSS':result[1],'CC':result[2],'SIM':result[3],'KLD':result[4]}
  print(path)
  print(my_result)
  
  final.append(my_result)
print('"""""""""""""""""""""""""""""""""""""""')  
print('final :')
AUC_Borji= sum([final[i]['AUC_Judd'] for i in range(len(lis_of_vid))])/len(lis_of_vid)
NSS= sum([final[i]['NSS'] for i in range(len(lis_of_vid))])/len(lis_of_vid)
CC= sum([final[i]['CC'] for i in range(len(lis_of_vid))])/len(lis_of_vid)
SIM= sum([final[i]['SIM'] for i in range(len(lis_of_vid))])/len(lis_of_vid)
KLD= sum([final[i]['KLD'] for i in range(len(lis_of_vid))])/len(lis_of_vid)
final_result ={'AUC_Judd':AUC_Judd,'NSS':NSS,'CC':CC,'SIM':SIM,'KLD':KLD}
print(final_result)