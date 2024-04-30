import numpy as np
import math as ma
import primme
from scipy.sparse import lil_matrix
from scipy.linalg import eig, inv
import matplotlib.pyplot as plt

def main():
  
  # constants
  n_max=6
  U = 1
  mu = np.sqrt(2)-1
  d = 2
  z = 2*d
  L_A=8

  CORTE = 60
  step = 0.01


  # Definition of Operators
  def Ham(phi):
    return -(b_op.todense() + b_op_tr.todense())*z*phi +  I.todense()*z*(phi**2) + (U/(2*J))*(n_sq-n_op.todense()) - (mu/J)*n_op.todense()

  def F_ab (alpha,beta,asterisco):
    psi_alpha=H_eigsh[1][:,alpha].reshape(n_max+1,1)
    psi_beta=H_eigsh[1][:,beta].reshape(n_max+1,1)
    psi_alpha=psi_alpha.reshape(1,n_max+1)
    if asterisco==False:
      return psi_alpha*b_op_tr.todense()*psi_beta
    if asterisco==True:
      return psi_beta.reshape(1,n_max+1)*b_op.todense()*psi_alpha.reshape(n_max+1,1)
    
  def EE(vn):
      return np.sum((1 + vn) * np.log(1 + vn) - vn * np.log(vn + 0.00001))
    
  n_op = lil_matrix((n_max+1,n_max+1))                                            
  for i in range (0,n_max+1):
    n_op[i,i]=i

  b_op = lil_matrix((n_max+1,n_max+1))
  for i in range (0,n_max+1):
    b_op[i,i-1]=np.sqrt(i)

  b_op_tr=b_op.transpose()
  I=lil_matrix((n_max+1,n_max+1))
  for i in range (0,n_max+1):
    I[i,i]=1
  n_sq=np.matmul(n_op.todense(),n_op.todense())

  GAMMA=lil_matrix((2*n_max,2*n_max))                                             
  for i in range(0,2*n_max):
    if i<n_max:
      GAMMA[i,i]=1
    else:
      GAMMA[i,i]=-1

  # Arrays for result storage
  ratio = []
  EE_array = []
  wa1 = []
  wa2 = []
  wa3 = []
  wa4 = []
  wa5 = []
  wa6 = []

  # Initialization for the run over k
  ee = 0
  wa1_n = 0
  wa2_n = 0
  wa3_n = 0
  wa4_n = 0
  wa5_n = 0
  wa6_n = 0

  #Run over interaction factor J
  for i_step in range(0, CORTE):

    J=(0.001+step*i_step)/z
    ratio.append(J*z / U)
    print('\n current: ', i_step, '/', CORTE - 1, ' '*5, 'J=',J*z/U)

    # Mean-Field selfconsistent solution 
    err=1
    phi=0.1
    while err>ma.pow(10,-6):                                                        
      H=Ham(phi)
      H_eigsh=primme.eigsh(H,k=n_max+1, which='SA')
      GS=H_eigsh[1][:,0].reshape(n_max+1,1)
      GS_T=GS.reshape(1,n_max+1)
      E_0=H_eigsh[0][0]
      phi_new=(GS_T*b_op.todense()*GS)[0,0]
      err=abs(phi_new-phi)
      phi=phi_new


    # Run over k-subspace
    ee = 0
    for i_k in range(0,int((int(L_A)+1))):
      k=(2*ma.pi/L_A)*((-L_A/2)+i_k)
      if k==0:
        break

      # Construction of quadratic Hamiltonian
      kx = ma.pi*0 
      eta_k=(ma.cos(k)+ma.cos(kx))/d

      A_0= lil_matrix((n_max,n_max))
      for i in range(0,n_max):
        A_0[i,i]=(H_eigsh[0][i+1]-E_0)*J

      A_1= lil_matrix((n_max,n_max))
      for i in range(0,n_max):
        for j in range(0,n_max):
          A_1[i,j]= -J* (F_ab(i+1,0,False)*F_ab(j+1,0,True) + F_ab(0,j+1,False)*F_ab(0,i+1,True) )

      B= lil_matrix((n_max,n_max))
      for i in range(0,n_max):
        for j in range(0,n_max):
          B[i,j]= -J* (F_ab(0,i+1,False)*F_ab(j+1,0,True)  +  F_ab(0,j+1,False)*F_ab(i+1,0,True))

      A_k=A_0+z*eta_k*A_1
      B_k=z*eta_k*B

      Matrix_AB= lil_matrix((2*n_max,2*n_max))                                   
      for i in range(0,2*n_max):
        for j in range(0,2*n_max):
          if i<n_max and j<n_max:
            Matrix_AB[i,j]=A_k[i,j]
          if i<n_max and j>=n_max:
            Matrix_AB[i,j]=B_k[i,j-n_max]
          if i>=n_max and j<n_max:
            Matrix_AB[i,j]=B_k[j,i-n_max]
          if i>=n_max and j>=n_max:
            Matrix_AB[i,j]=A_k[j-n_max,i-n_max]

      # Construction of Correlation Matrix
      Y = np.block([[np.eye(n_max), np.zeros((n_max, n_max))], [np.zeros((n_max, n_max)), -np.eye(n_max)]])
      MM = np.dot(Y, Matrix_AB.todense())
      Qo = np.linalg.eig(MM)[1]
      sp = np.argsort(np.diag(    np.dot( np.linalg.inv(Qo) , np.dot( Y ,  np.dot(  Matrix_AB.todense() , Qo)))    ))
      lv = np.concatenate((sp[n_max:], sp[:n_max][::-1]))
      QQBO = Qo[:, lv]
      Spectrum = np.diag((    np.dot( np.linalg.inv(QQBO) , np.dot( Y ,  np.dot(  Matrix_AB.todense() , QQBO)))    ))
      Omega = (    np.dot( np.linalg.inv(QQBO) , np.dot( Y ,  np.dot(  Matrix_AB.todense() , QQBO)))    )
      MP = np.block([[np.eye(n_max), np.zeros((n_max, n_max))], [np.zeros((n_max, n_max)), np.zeros((n_max, n_max))]])
      CC = np.dot(np.dot(QQBO, MP), QQBO.transpose())
      lc, qc = np.linalg.eig(np.dot(Y, CC))
      QCo = qc.transpose()
      vn = (np.sort(lc)[n_max:2*n_max]-1)

      wa1_n += np.log(1+1/(abs(vn[0])+0.00001))
      wa2_n += np.log(1+1/(abs(vn[1])+0.00001))
      wa3_n += np.log(1+1/(abs(vn[2])+0.00001))
      wa4_n += np.log(1+1/(abs(vn[3])+0.00001))
      wa5_n += np.log(1+1/(abs(vn[4])+0.00001))
      wa6_n += np.log(1+1/(abs(vn[5])+0.00001))

      #Entanglement Entropy for specific k
      ee+=EE(abs(vn))

    wa1.append(wa1_n)
    wa2.append(wa2_n)
    wa3.append(wa3_n)
    wa4.append(wa4_n)
    wa5.append(wa5_n)
    wa6.append(wa6_n)

    #Entanglement Entropy as f(J)
    EE_array.append(ee)


  # Plot Results
  plt.figure(figsize=(10,10))
  plt.plot(ratio, EE_array, 'b', label='$S$', marker='.', linestyle='-')
  plt.xlabel('$J z /U$',fontsize=25)
  plt.ylabel('$S_{ent}$',fontsize=35)
  plt.yticks(fontsize=13)
  plt.xticks(fontsize=13)
  #plt.ylim(bottom=0, top=18.5)
  plt.title('')
  axes = plt.gca()
  plt.legend(fontsize=13)
  plt.grid()
  plt.text(0.08, 0.9, '$L_A=$'+str(L_A), transform=plt.gca().transAxes, fontsize=20, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
  if mu!=np.sqrt(2)-1:
    plt.text(0.08, 0.80, '$\mu=$'+str(mu), transform=plt.gca().transAxes, fontsize=20, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
  else:
    plt.text(0.08, 0.80, '$\mu=\sqrt{2}-1$', transform=plt.gca().transAxes, fontsize=20, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
  plt.show()


if __name__ == "__main__":
    main()