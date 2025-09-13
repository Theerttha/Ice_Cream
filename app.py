import matplotlib.pyplot as pt
import matplotlib.ticker as mticker
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
def open_file():
    df = pandas.read_csv("IceCreamData.csv")
    x=df["Temperature"].to_numpy().reshape(-1,1)
    y=df["Revenue"].to_numpy().reshape(-1,1)
    return df,x,y
def plot(x,y):
   

    pt.xlabel("Temperature in \u00b0C")
    pt.ylabel("Revenue in $")
    
    values=stats.linregress(x, y)
    pt.scatter(x,y,color="#89BAC1")

    pt.plot(x,(values.slope*x+values.intercept),color="#346E43")
    pt.show()
def linear_regression(X,Y):
    clf = LinearRegression()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    clf.fit(X_train, Y_train)
    prediction=clf.predict(X_test)
    score=clf.score(X_test,Y_test)
    print(score)
    return X_test,Y_test,prediction
def plot_prediction(x,y,prediction):
    pt.xlabel("Temperature in \u00b0C")
    pt.ylabel("Revenue in $")
    print(len(x))
    pt.scatter(x,y,color="#89BAC1",label="Actual")
    pt.scatter(x,prediction,color="#346E43", label="Predicted")
    pt.legend(loc="right")
    pt.show()
if "__main__" == __name__:
    df,x,y= open_file()
    x_test,y_test, prediction=linear_regression(x,y)
    plot(x,y)
    plot_prediction(x_test,y_test,prediction)