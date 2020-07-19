import numpy as np

class RBF():
    def __init__(self,n_clusters, epochs=100):
        self.n_clusters = n_clusters
        self.w = []
        self.error = []
        self.h = 0
        self.centers = []
    def gaussian(self,x,media):
        d_aux = 0
        for d in range(len(x)):
            d_aux += pow(x[d]-media[d],2)
        distancia = np.sqrt(d_aux)
        return (np.exp(distancia/(self.h**2)))
    
    def H(self,centers):
        d_media = 0 
        for i in range(len(centers)):
            for j in range(len(centers)-1):
                d_aux = 0
                for d in range(len(centers[0])):
                    d_aux += pow(centers[i][d]-centers[j][d],2)
                d_media += np.sqrt(d_aux)
        d_media = d_media/len(centers)*(len(centers)-1)
        return d_media/2
    def escolha_dos_centros(self,n_clusters,x_data):
        centers = []
        #print(n_clusters)
        p = [0]*n_clusters
        #seleciona os clusters aleatoriamente
        for i in range(n_clusters):
            p[i] = np.random.randint(0,len(x_data)-1,1)[0]
            centers.append(x_data[p[i]])
        self.centers = centers
        return centers
    def fit(self,x_data,y_data):
        x_data,y_data = np.array(x_data),np.array(y_data)
        testMatrixSingular = True
        while testMatrixSingular:
            #Parte I
            centers = self.escolha_dos_centros(self.n_clusters,x_data)
            #print(centers)
            #parte II
            #calcular paramentro h
            self.h = self.H(centers)
            #parte III
            x_new = []
            for i in range(len(x_data)):
                x_aux = []
                for j in range(len(centers)):
                    #print(x_data[i],centers[j])
                    x_aux.append(self.gaussian(x_data[i],centers[j]))
                x_new.append(x_aux)
            #calculo dos pesos 
            x_new  =  np.array(x_new)

            try:
                w = np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T).dot(y_data)
                self.w = w
                testMatrixSingular = False
            except np.linalg.LinAlgError:
                testMatrixSingular = True
                #print('matriz singular')  
        
    def predict(self,x):
        y_pred = []
        for i in range(len(x)):
            x_new = []
            for j in range(self.n_clusters):
                x_new.append(self.gaussian(x[i],self.centers[j]))
            x_new = np.array(x_new)
            y = x_new.dot(self.w)
            if y<0: y=0
            y_pred.append(int(round(y)))
        return y_pred
