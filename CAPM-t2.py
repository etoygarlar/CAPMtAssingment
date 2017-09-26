# -*- coding: utf-8 -*-
"""
CAPM-t2.py

Purpose:
    Calculate the excess returns of stocks 'IBM', 'F', 'XOM', 'AIG' AGAINST 'DTB3'
    To have an OLS on these stocks against index SP500

Version:
    For Python 2, using np.dot for matrix multiplication

Date:
    2017/9/14

@author: etr430 (Elvan Toygarlar) & pbs218 (Pim Burgers)
"""

import numdifftools as nd
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import t


def _gh_stepsize(vP):
    """
    Purpose:
        Calculate stepsize close (but not too close) to machine precision

    Inputs:
        vP      1D array of parameters

    Return value:
        vh      1D array of step sizes
    """
    vh = 1e-8*(np.fabs(vP)+1e-8)   # Find stepsize
    vh = np.maximum(vh, 5e-6)       # Don't go too small

    return vh


def hessian_2sided(fun, vP, *args):
    """
    Purpose:
      Compute numerical hessian, using a 2-sided numerical difference

    Author:
      Kevin Sheppard, adapted by Charles Bos

    Source:
      https://www.kevinsheppard.com/Python_for_Econometrics

    Inputs:
      fun     function, as used for minimize()
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      mH      iP x iP matrix with symmetric hessian
    """
    iP = np.size(vP,0)
    vP = vP.reshape(iP)    # Ensure vP is 1D-array

    f = fun(vP, *args)
    vh = _gh_stepsize(vP)
    vPh = vP + vh
    vh = vPh - vP

    mh = np.diag(vh)            # Build a diagonal matrix out of vh

    fp = np.zeros(iP)
    fm = np.zeros(iP)
    for i in range(iP):
        fp[i] = fun(vP+mh[i], *args)
        fm[i] = fun(vP-mh[i], *args)

    fpp = np.zeros((iP,iP))
    fmm = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            fpp[i,j] = fun(vP + mh[i] + mh[j], *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(vP - mh[i] - mh[j], *args)
            fmm[j,i] = fmm[i,j]
    
    vh = vh.reshape((iP,1))
    mhh = np.dot(vh, vh.T)             # mhh= h h', outer product of h-vector

    mH = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            mH[i,j] = (fpp[i,j] - fp[i] - fp[j] + f + f - fm[i] - fm[j] + fmm[i,j])/mhh[i,j]/2
            mH[j,i] = mH[i,j]

    return mH


def create_data(stocks, path):

    """
        data= create_data(stocks,path)

        Purpose:
          Create a dataframe with data downloaded from yahoo finance and remove unmatched indexes

        Inputs:
          stocks    list of stocks we use
          path      directory of data from yahoo finance

        Outputs:
          data      [(Size(stocks)+2) x len(data)] dataframe of prices of stocks + SP500 + DTB3 (Bond)

        Return value:
          data      [(Size(stocks)+2) x len(data)] dataframe of stocks + SP500 + DTB3 (Bond)

    """

    data = pd.DataFrame()
    for symbol in stocks:
        data[symbol] = pd.read_csv('%s/%s.csv' %(path, symbol), index_col="Date" )['Adj Close']
    data['BOND'] = pd.read_csv('%s/DTB3.csv' % path, index_col="DATE", na_values=['.'])['DTB3']
    data.BOND = data.BOND.astype(np.float)
    data['SP500'] = pd.read_csv('%s/GSPC.csv' % path, index_col="Date")['Adj Close']
    data.index = pd.to_datetime(data.index)

    # eliminate nans:
    data = data[(~data.BOND.isnull()) & (~data.SP500.isnull())]

    return data


def Excess_logRet(data, stocks, index, is_monthly=False):

    """
        data= Excess_logRet(data, stocks, index, dm)

        Purpose:
          First calculate the log return of each stock and then
          calculate their excess return against the bond return.

        Inputs:
          data    dataframe of stocks + SP500 + DTB3 (Bond)
          stocks  list of stocks we use
          index   string which is name of the index (SP500)
          is_monthly boolean decides to take excess return of daily or monthly data (by default it is False)

        Outputs:
          data      [((Size(stocks) + 1 ) * 4 + 1) x len(data)] dataframe
                    prices of stocks + SP500 + DTB3 (Bond)
                    and log return, excess return of stocks + SP500

        Return value:
          data      dataframe of prices of stocks + SP500 + DTB3 (Bond) and log return, excess return
                    of stocks + SP500
    """

    # get the logs of stocks and index
    for s in stocks + [index]:
        data['log_%s' % s] = np.log(data[s])
        data['return_%s' % s] = 100 * data['log_%s' % s].diff()
        if is_monthly:
            data['ex_return_%s' % s] = data['return_%s' % s] - data.BOND / 250
        else:
            data['ex_return_%s' % s] = data['return_%s' % s] - data.BOND / 12

    data = data.iloc[1:]

    return data


def normal_loglikehood(theta, x, y):
    """
    Purpose: 
            Computing the log likelihood of a normal distribution
    Inputs: 
            Theta: vector with elements sigma and beta
            x: iN x 1 vector of market index returns
            y: iN x 1 vector of stock index returns
    Return Value:
            Log_probabilities: iN x 1 vector of log likelihood 
    """
    
    beta0 = theta[1]
    beta1 = theta[2]
    sigma = theta[0]
    
    vE = y - np.dot(x, beta1)-beta0  

    log_probabilities = -0.5 * (np.log(2 * np.pi * sigma * sigma) + np.square(vE / sigma))

    return log_probabilities


def StudentT_loglikehood_func(thetaT, x, y):
    """
    Purpose: 
            Computing the log likelihood of a t distribution
    Inputs: 
            ThetaT: vector with elements sigma, beta and nu
            x: iN x 1 vector of market index returns
            y: iN x 1 vector of stock index returns
    Return Value:
            Log_probabilities: iN x 1 vector of log likelihood 
    """
    beta0 = thetaT[1]
    beta1 = thetaT[2]
    sigma = thetaT[0]
    dF = thetaT[3]
    
    vE = y - np.dot(x, beta1)-beta0
    
    log_probabilities = t.logpdf(vE, df=dF, scale=np.sqrt(sigma))
    
    return log_probabilities
    

def main():
    # Magic Numbers
    path = '.'
    stocks = ['IBM']
    index = 'SP500'


    data = create_data(stocks, path)
    
    data = Excess_logRet(data, stocks, index, 'd')
    
    theta = np.array([0.5, 1, 1])
    thetaT = np.array([0.5, 1, 1, 3])
    
    # Initialising
    y = np.array(data['ex_return_IBM'])
    x = np.array(data['ex_return_SP500'])
    
    # To check whether it works for uniform distribution
    #x=np.random.uniform(0,1,4400)
    #y=np.random.uniform(0,1,4400)

    iN = np.size(x)
    x = x.reshape((iN, 1))
    y = y.reshape((iN, 1))
    
    # Estimation    
    # for T distribution
    LLT = lambda thetaT: -np.mean(StudentT_loglikehood_func(thetaT, x, y), axis=0)    
    LL = opt.minimize(LLT, thetaT)    
    
    mH = -hessian_2sided(LLT, thetaT)
    mHI = np.linalg.inv(mH)/iN
    mHI = (mHI + mHI.T)/2    
    
    fnSc = nd.Jacobian(StudentT_loglikehood_func)
    mG = fnSc(thetaT, y, x)
    mS2emp = -mHI
    mS2rob = np.dot(np.dot(mHI,np.dot(mG.T,mG)), mHI)
    
    # Output of T-distribution
    print('\n -Hessian inv for t-dist ML:\n\n %s\n' % mS2emp)
    print ('Robust sandwich form for t distribution:\n\n%s\n' % mS2rob)
    LL = LL.x
    print('\n Result of the t-dist ML : %s' % LL)
    
    # Estimation 
    # for Normal distribution
    LLN = lambda theta: -np.mean(normal_loglikehood(theta, x, y), axis=0)  
    LLNN = opt.minimize(LLN, theta)
    
    mHN = -hessian_2sided(LLN, theta)
    mHNI = np.linalg.inv(mHN)/iN
    mHNI = (mHNI + mHNI.T)/2    
    
    fnSc = nd.Jacobian(normal_loglikehood)
    mGN = fnSc(theta, y, x)
    mS2Nemp = -mHNI
    mSN2rob = np.dot(np.dot(mHNI, np.dot(mGN.T, mGN)), mHNI)
    
    # Output for normal distribution
    print('\n -Hessian inv for normal ML:\n\n %s\n' % mS2Nemp)
    print ('Robust sandwich form Normal distribution:\n\n %s\n' % mSN2rob)
    LLNN = LLNN.x
    print('\n Result of the normal ML : %s ' % LLNN)
    




if __name__ == "__main__":
    main()