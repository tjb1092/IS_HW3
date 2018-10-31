import numpy as np
from data_load import preprocessData, create_train_test_split
import math
class NN:
    def __init__(self, n, H, LR, f, c):
        self.n = n
        self.f = f
        self.c = c
        self.H = H
        self.LR = LR
        layers = []
        for L in range(self.n+1):
            # Get dims for weight matrix w/ dims i,j
            if L == 0:
                i = self.H[L]
                j = self.f + 1
                sigma = math.sqrt(2/(f+1))
            elif L == n:
                i = c
                j = self.H[L-1]+1
                sigma = math.sqrt(2/(H[L-1]+1))
            else:
                i = self.H[L]
                j = self.H[L-1]+1
                sigma = math.sqrt(2/(H[L-1]+1))


            w_ij = np.random.normal(0,sigma,(i,j))
            print("Layer {}".format(L))
            print(w_ij)
            print(w_ij.shape)
            layers.append(w_ij)

        self.layers = layers

    def fit(self, X_q,y_q):
        #Forward and backward prop on one data point. Return error matricies
        # Assume sigmoid activation functions for now.

        #forward pass
        f_s_aq=[]
        for L in range(self.n):
            w_bar = self.layers[L]
            # Set the bias to 1, will override rest of values.
            h_q = np.ones((w_bar.shape[0]+1,1))
            if L==0:
                #print("X")
                #print(X_q)

                dot_p = np.dot(w_bar,X_q)


            else:
                dot_p = np.dot(w_bar,f_s_aq[L-1])
                #print("in")
                #print(f_s_aq[L-1])
            #print("W")
            #print(w_bar)
            #print("dotp")
            #print(dot_p)
            h_q[1:] = 1./(1.+np.exp(-dot_p))  # sigmoid activation
            #print("h_q")
            #print(h_q)
            f_s_aq.append(h_q)

        dot_p = np.dot(self.layers[self.n],f_s_aq[self.n-1])
        #print("in")
        #print(f_s_aq[self.n-1])
        #print(self.layers[self.n])
        y_hatq = 1./(1+np.exp(-dot_p))
        #print("dotp")
        #print(dot_p)
        #print("yhat")
        #print(y_hatq)
        f_s_aq.append(y_hatq)
        print("predictions")
        print(y_hatq)
        #print(f_s_aq)
        print("Forward Pass Complete")

        # Backward pass: Compute output layer's errors
        # Initialize using list comprehension
        delq = [[] for k in range(self.n+1)]
        delW = [[] for k in range(self.n+1)]
        delq[-1] = (1-y_hatq)*y_hatq*(y_q - y_hatq)
        #print("delq")
        #print(delq[-1])
        #print("in")
        #print(f_s_aq[self.n-1].T)
        #delW= eta(L)*deltaq(L)*h(L-1)
        delW[-1] = self.LR[self.n] * np.dot(delq[-1], f_s_aq[self.n-1].T)
        #print("delW")
        #print(delW[-1])
        #input("pasue")

        for L in range(self.n-1, -1, -1):
            # Next layer:
            #print("L", L)
            # h[L] ^* W*delq[L+1]
            delq[L] = ((1-f_s_aq[L])*f_s_aq[L])[1:]*np.dot(self.layers[L+1][:,1:].T,delq[L+1])
            if L==0:
                inputQ = X_q.T
            else:
                inputQ = f_s_aq[L-1].T

            #print("in")
            #print(inputQ)
            #print(delq[L])

            delW[L] = self.LR[L]*np.dot(delq[L],inputQ)
            #print(delW[L], delW[L].shape)
            #input("pasue")

        #print("delW")
        #print(delW)
        #print("W")
        #print(self.layers)
        # Apply weight update after data point has been applied
        for i, W in enumerate(self.layers):
            self.layers[i] = W + delW[i]
        #print("after update")
        #print(self.layers)
def main():
    nn = NN(2, [2, 2], [0.1, 0.1, 0.1], 2, 2)

    X, y = data_load()
    epochs = 5000
    for epoch in range(epochs):
        X = np.array((1,1))
        X = np.insert(X, 0, 1).reshape(X.shape[0]+1,1)  # Add bias term
        y = np.array((0,1)).reshape(2,1)
        nn.fit(X,y)

        X = np.array((-1, -1))
        X = np.insert(X, 0, 1).reshape(X.shape[0]+1,1)  # Add bias term
        y = np.array((1, 0)).reshape(2,1)
        nn.fit(X,y)




if __name__ == "__main__":
    main()
