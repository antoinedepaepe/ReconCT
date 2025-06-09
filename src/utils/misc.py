from src.operators.operator import Operator
import numpy as np
import matplotlib.pyplot as plt
import torch

def check_transpose(A:Operator,N:int,Ni:int,istr:bool):
 
 x0 = np.random.randn(N,N) 
 if istr==True:
    device='cuda'
    x0 = np.float32(x0) 
    x0 = torch.from_numpy(x0)
    x0 = x0.to(device)
 
 y0 = A.transform(x0)
 if istr==True:
    y0 = y0.squeeze().cpu()

 dimx = x0.shape
 dimy = y0.shape




 a_list = []
 b_list = []


 for k in range(Ni):
  x = np.random.randn(dimx[0],dimx[1]) 
  y = np.random.randn(dimy[0],dimy[1]) 
  
  if istr==True:
    x = np.float32(x)  
    x = torch.from_numpy(x)
    x = x.to(device)
    y = np.float32(y) 
    y = torch.from_numpy(y)
    y = y.to(device)
  
  Ax = A.transform(x)
  ATy = A.transposed_transform(y)
  
  if istr==True:
    x = x.squeeze().cpu()
    y = y.squeeze().cpu()
    Ax = Ax.squeeze().cpu()
    ATy = ATy.squeeze().cpu()
  
  
  a = np.dot(Ax.ravel(), y.ravel())
  b = np.dot(x.ravel(), ATy.ravel())
  a_list.append(a)
  b_list.append(b)
  print(a)
  print(b)

 # now make the scatter plot
 plt.figure(figsize=(6,6))
 plt.scatter(a_list, b_list, alpha=0.6, edgecolor='k')
 # plot the identity line
 # v_min = min(min(a_list), min(b_list))
 # v_max = max(max(a_list), max(b_list))
 # plt.plot([v_min, v_max], [v_min, v_max], 'r--', lw=1)
 plt.xlabel('a = ⟨A x, y⟩')
 plt.ylabel('b = ⟨x, Aᵀ y⟩')
 plt.title(f'Check transpose property over {Ni} trials')
 # plt.axis('equal')   # square axes so the identity line is at 45°
 plt.show() 
 scales = [a/b for a,b in zip(a_list, b_list) if b!=0]
 print("scale ≃", np.median(scales)) 
 
