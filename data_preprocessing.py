import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d import Axes3D
import datetime
from sklearn import model_selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor



#Split Data
seed = 523
X_train, X_val_test, Y_train_sntr, Y_val_test_sntr, Y_train_dnp, Y_val_test_dnp, Y_train_total, Y_val_test_total = model_selection.train_test_split(X, Y_sntr, Y_dnp, Y_total, test_size= 0.2, shuffle=True, random_state = seed)
seed = 768
X_val, X_test, Y_val_sentr, Y_test_sentr, Y_val_dnp, Y_test_dnp, Y_val_total, Y_test_total = model_selection.train_test_split(X_val_test, Y_val_test_sentr, Y_val_test_dnp, Y_val_test_total, test_size= 0.5, shuffle=True,
random_state = seed)
#Ratio = 0.8 , 0.1 , 0.1


#Models
different_predictions = [sntr, dnp, total]
for p in different_predictions:
    if p == sentr:
        Y_train = Y_train_sentr
        Y_val = Y_val_sentr
    elif p == dnp:
        Y_train = Y_train_dnp
        Y_val = Y_val_dnp
    else:
        Y_train = Y_train_total
        Y_val = Y_val_total
    
    #Linear Regression with polynomial basic function
    mse_train = []
    mse_val = []
    r2_train = []
    r2_val = []

    for k in range(6):
        poly_model = make_pipeline(PolynomialFeatures(k),LinearRegression())
        poly_model.fit(X_train, Y_train)

        Y_train_pred = poly_model.predict(X_train)
        Y_val_pred = model.predict(X_val)

        mse_train.append(mean_squared_error(Y_train, Y_train_pred))
        r2_train.append(r2_score(Y_train, Y_train_pred))
        mse_val.append(mean_squared_error(Y_val, Y_val_pred))
        r2_val.append(r2_score(Y_val, Y_val_pred))

    plt.figure(figsize=(10,10))
    plt.ylim(min(min(mse_train),min(mse_val)), min(max(max(mse_train), max(mse_val)), 300))
    plt.plot(x_axis, mse_train, label="Training")
    plt.plot(x_axis, mse_val, label="Validation")
    plt.title(label= p + ": mse: Linear Regression with polynomial basic function of degree "+str(k))
    plt.legend()

    plt.figure(figsize=(10,10))
    plt.ylim(0,1)
    plt.plot(range(6), r2_train, label="Training")
    plt.plot(range(6), r2_val, label="Validation")
    plt.title(label= p + ": r2_score: Linear Regression with polynomial basic function of degree "+str(k))
    plt.legend()




    #MLP-Regressor
    #Testing for different values of hiddenlayers, alpha, learningrate
    #Honestly dont really know what are realistic values
    alphalist = [0.001, 0.01, 0.1]
    learningratelist = [0.001, 0.01, 0.1]
    mse_train = np.ones(len(learningratelist),len(alphalist))
    mse_val = np.ones(len(learningratelist),len(alphalist))
    r2_train = np.ones(len(learningratelist),len(alphalist))
    r2_val = np.ones(len(learningratelist),len(alphalist))
    i,j = 0
    for hiddenlayers in [10,100]:
        for learning_rate in learningratelist:
            for alpha in alphalist:

                mlp_reg = MLPRegressor(hidden_layer_sizes=(hiddenlayers,), learning_rate ="constant",
                learning_rate_init = learningrate, alpha = alpha)

                mlp_reg.fit(X_train, Y_train)
                Y_train_pred = mlp_reg.predict(X_train)
                Y_val_pred = mlp_reg.predict(X_val)

                mse_train[i][j]=mean_squared_error(Y_train_pred, Y_train)
                mse_val[i][j]=mean_squared_error(Y_val_pred, Y_val)
                r2_train[i][j]=r2_score(Y_train_pred, Y_train)
                r2_train[i][j]=r2_score(Y_val_pred, Y_val)

                j+=1
            i+=1

        learning_mesh, alpha_mesh = np.meshgrid(learningratelist, alpharatelist)
        mse_train_mesh = np.meshgrid(mse_train)
        mse_val_mesh =np.meshgrid(mse_val)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_wireframe(learning_mesh,alpha_mesh, mse_train_mesh, color = "red")
        ax.plot_wireframe(learning_mesh, alpha_mesh, mse_val_mesh, color = "black")
        ax.set_title(p+": MSE: MLP using "+str(hiddenlayers)+ " hidden layers")

        r2_train_mesh = np.meshgrid(r2_train)
        r2_val_mesh =np.meshgrid(r2_val)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_wireframe(learning_mesh,alpha_mesh, r2_train_mesh, color = "red")
        ax.plot_wireframe(learning_mesh, alpha_mesh, r2_val_mesh, color = "black")
        ax.set_title(p+": r2_score: MLP using "+str(hiddenlayers)+ " hidden layers")




    #K-nn Regressor
    mse_train = []
    mse_val = []
    r2_train = []
    r2_val = []

    for k in range(1,10):
        knn = KNeighborsRegressor(n_neighbors = k, p=2)
        knn.fit()
        Y_train_pred = knn.predict(Y_train)
        Y_val_pred = knn.predict(Y_val)

        mse_train.append(mean_squared_error(Y_train, Y_train_pred))
        mse_val.append(mean_squared_error(Y_val, Y_val_pred))
        r2_train.append(r2_score(Y_train, Y_train_pred))
        r2_val.append(r2_score(Y_val, Y_val_pred))

    plt.figure(figsize=(10,10))
    plt.ylim(min(min(mse_train),min(mse_val)), min(max(max(mse_train), max(mse_val)), 300))
    plt.plot(x_axis, mse_train, label="Training")
    plt.plot(x_axis, mse_val, label="Validation")
    plt.title(label= p + ": mse: knn with "+str(k)+" neighbors")
    plt.legend()

    plt.figure(figsize=(10,10))
    plt.ylim(0,1)
    plt.plot(range(6), r2_train, label="Training")
    plt.plot(range(6), r2_val, label="Validation")
    plt.title(label= p + ": r2_score: knn with "+str(k)+" neighbors")
    plt.legend()