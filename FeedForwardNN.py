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
            print(w_ij)
            print(w_ij.shape)
            layers.append(w_ij)

        self.layers = layers

    def fit(self, X_q,y_q):
        #Forward and backward prop on one data point. Return error matricies
        # Assume sigmoid activation functions for now.

        #forward pass
        f_s_aq=[]
        #s_aq = []
        for L in range(self.n):
            w_bar = self.layers[L]
            # Set the bias to 1, will override rest of values.
            h_q = np.ones((w_bar.shape[0]+1,1))
            if L==0:
                dot_p = np.dot(w_bar,X_q)

            else:
                dot_p = np.dot(w_bar,f_s_aq[L-1])
            h_q[1:] = 1./(1.+np.exp(-dot_p))  # sigmoid activation
            print(h_q)
            input("pause")
            f_s_aq.append(h_q)
            #s_aq.append(dot_p)

        dot_p = np.dot(self.layers[self.n],f_s_aq[self.n-1])
        y_hatq = 1./(1+np.exp(-dot_p))
        print(y_hatq)
        input("pasue")
        f_s_aq.append(y_hatq)
        #s_aq.append(dot_p)
        print("Forward Pass Complete")

        # Backward pass ( not currently modular)

        #Compute output layer's errors
        del_nq = (1-y_hatq)*y_hatq*(y_q - y_hatq)
        #delW= eta(L)*deltaq(L)*h(L-1)
        delWn = self.LR[self.n] * np.dot(del_nq, f_s_aq[self.n-1].T)

        print(delWn)
        input("pasue")
        # Next layer:
        del_1q = ((1-f_s_aq[self.n-1])*f_s_aq[self.n-1])[1:]*np.sum(del_nq*self.layers[self.n], axis=1).reshape(del_nq.shape)
        #del_1q = del_1q.reshape(len(del_1q),1)
        delW1 = self.LR[self.n-1]*np.dot(del_1q, f_s_aq[self.n-2].T)
        print(delW1, delW1.shape)
        input("pasue")

        # Final layer:
        del_0q = ((1-f_s_aq[self.n-2])*f_s_aq[self.n-2])[1:]*np.sum(del_1q*self.layers[self.n-1], axis=1).reshape(del_1q.shape)
        #del_1q = del_1q.reshape(len(del_1q),1)
        delW0 = self.LR[self.n-2]*np.dot(del_0q, X_q.T)
        print(delW0, delW0.shape)
        input("pasue")




def main():
    nn = NN(2, [2,2], [0.001, 0.001, 0.001], 3, 2)
#
    X = np.array((0,1,2))
    X = np.insert(X, 0, 1).reshape(X.shape[0]+1,1)  # Add bias term
    y = np.array((0,1)).reshape(2,1)
    nn.fit(X,y)


if __name__ == "__main__":
    main()
