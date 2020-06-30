from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 

'''This is code for simple gaussian process regression. It assumes a zero mean GP Prior
'''

# Ground truth function
f = lambda x: np.sin(0.9*x).flatten()
f1 = lambda x: (0.25*(x**2)).flatten()

# Kernel function
def kernel(a,b):
    ''' GP squared exponential kernel/Gaussian Kernel Function
    '''
    sigma = 0.1 # parameter
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a,b.T)
    return np.exp(-.5*(1/sigma)*sqdist)
    
class GausianProcessRegression():
    ''' Gausian Process for linear regression
    '''
    def __init__(self, kernel_ = None):
        if kernel_ is None:
            self.K = kernel
        else:
            self.K = kernel
        self.is_fitted = False 
        
    def fit(self,X,Y):
        '''Train model'''
        N = X.shape[0]
        # Get cholesky decomposition (square root) of the
        # covariance matrix
        self.L = np.linalg.cholesky(self.K(X,X)+s*np.eye(N))
        self.X = X
        self.Y = Y
        self.is_fitted = True
        
    def predict(self,Xtest):
        '''Test model'''
        if not self.is_fitted:
            print("Please fit the model before predict")
            return
        #Compute the mean at our test points
        LK = np.linalg.solve(self.L,self.K(self.X,Xtest))
        mu = np.dot(LK.T, np.linalg.solve(self.L,self.Y))
        #Compute the variance at our test points.
        K_ = self.K(Xtest, Xtest)
        s2 = np.diag(K_) - np.sum(LK**2, axis = 0)
        s = np.sqrt(s2)
        
        return mu, s 
     
if __name__ == "__main__":
    '''Example code can run here'''
    print('*'*5 + "Gaussian Process" + '*'*5)
    
    # Generate training data 
    N = 10 # number of training points
    n = 50 # number of test points.
    s = 0.00005 # noise variance
    X_train = np.random.uniform(-5,5,size=(N,1))
    Y_train = f(X_train) + s*np.random.randn(N)
    
    # Generate testing data 
    X_test = np.linspace(-5,5,n).reshape(-1,1)
    
    # Train model 
    GP = GausianProcessRegression()
    GP.fit(X_train,Y_train)
    
    # Test model 
    mu,sigma = GP.predict(X_test)
     
    #print(mu)
    #print(sigma)
    
    # Plot ground true 
    #X = np.linspace(-5,5,100).reshape(-1,1)
    #Y = f(X)
    #Y1 = f1(X)
    #plt.figure(1)
    #plt.clf()
    #plt.plot(X,Y,'r-')
    #plt.plot(X,Y1,'b+')
    #plt.legend(['GT','F1'])
    #plt.title('Ground true function')
    #plt.axis([-5,5,-3,3])
    
    # PLOTS:
    plt.figure(1)
    plt.clf()
    plt.plot(X_train,Y_train,'r+',ms=20)
    plt.plot(X_test,f(X_test),'b-')
    plt.gca().fill_between(X_test.flat, mu -3*sigma, mu+3*sigma, color="#dddddd")
    plt.plot(X_test,mu, 'g--',lw=2)
    plt.title("Mean predictions plus 3 standard deviations")
    plt.axis([-5,5,-3,3])
    plt.savefig('predictive.png', bbox_inches='tight')
    
    # draw samples from the prior at our test points.
    K_ = kernel(X_test, X_test)
    L = np.linalg.cholesky(K_ + s*np.eye(n))
    f_prior = np.dot(L,np.random.normal(size=(n,10)))
    plt.figure(2)
    plt.clf()
    plt.plot(X_test,f_prior)
    plt.title('Ten samples from the GP prior')
    plt.axis([-5,5,-3,3])
    plt.savefig('prior.png', bbox_inches='tight')
    
    plt.show()