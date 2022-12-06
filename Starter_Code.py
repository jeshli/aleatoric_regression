#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:51:36 2022

@author: jeshli
"""



import pandas as pd
import torch
import numpy as np
import Aleatoric_Model
import matplotlib.pyplot as plt



Synthetic_Demo = True

if Synthetic_Demo:
    DF = pd.DataFrame()
    Time_Data = pd.date_range(start='2021-05-21', end='today')
    DF['Date'] = Time_Data
    t = ((Time_Data - Time_Data[0]).days / 365).values
    s = np.linspace(1e-1,2e-1,num=t.shape[0])
    y = np.random.normal(t,s)
    y /= y.std()
    log_Cycle_Burn_Data = y
    Cycle_Burn_Data = np.exp(log_Cycle_Burn_Data)
    DF['Cycles Burn'] = Cycle_Burn_Data


###############################################################################



Cycle_Burn_Data = DF['Cycles Burn'].values
Time_Data = DF['Date']
t = ((Time_Data - Time_Data[0]).dt.days / 365).values
log_Cycle_Burn_Data  = np.log(Cycle_Burn_Data)



y = log_Cycle_Burn_Data
Y = torch.from_numpy(y).float().unsqueeze(-1)
X = torch.from_numpy(t).float().unsqueeze(-1)



AM = Aleatoric_Model.Aleatoric_Regression_Model()
AM.fit(X,Y,exponential=False,verbose=True,lr=1e-2,epochs=3000,show_at=500)
u,s = AM.predict(torch.from_numpy(t).float().unsqueeze(-1))


plt.plot(Time_Data,y)
plt.plot(Time_Data,u.detach().numpy(), color='r')
plt.plot(Time_Data,u.detach().numpy() + 2 * s.detach().numpy(),color='orange')
plt.plot(Time_Data,u.detach().numpy() - 2 * s.detach().numpy(),color='orange')
plt.title('Log Space - Modeled Burn Rate Prediction')
plt.xlabel('Date')
plt.xticks(rotation = 45) 
plt.ylabel('Log Cycles Burned')
plt.legend(['data','mean','2 sigma'])
plt.show()


plt.plot(Time_Data,Cycle_Burn_Data)
plt.plot(Time_Data,np.exp(u.detach().numpy()), color='r')
plt.plot(Time_Data,np.exp(u.detach().numpy() + 2 * s.detach().numpy()),color='orange')
plt.plot(Time_Data,np.exp(u.detach().numpy() - 2 * s.detach().numpy()),color='orange')
plt.title('Modeled Burn Rate Prediction')
plt.xlabel('Date')
plt.xticks(rotation = 45) 
plt.ylabel('Cycles Burned')
plt.legend(['data','mean','2 sigma'])
plt.show()


print('Exponential Rate of Increase YoY', str( np.round(100 * AM.model.mu.weight.item(),2) ) + '%')
print('Variance Rate of Increase YoY', str( np.round(100 * AM.model.sigma.weight.item(),2) ) + '%')