from masonite.request import Request
from masonite.filesystem import Storage
from masonite.views import View
from masonite.controllers import Controller
from masonite.validation import Validator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class SimpleLinearRegressionController(Controller):
    def show(self, view: View):
        return view.render("upload")

    def upload(self, request: Request, view: View, validate: Validator, storage: Storage):
        errors = request.validate(
            {
                "dataset": "required|file",
            }
        )

        if errors:
            return view.render("upload", {"errors": errors})

        dataset = request.input("dataset")
        file_path = os.path.join("storage", "uploads", dataset.filename)
        target_dir = os.path.join("storage", "uploads", dataset.filename)
        #disk = os.path.join(storage.disk('local').path, "storage", "uploads")
        print(dataset.filename)
        storage.disk('local').put_file(target_dir, dataset, dataset.filename)

        df = pd.read_csv(file_path)

        columns = df.columns.tolist()

        return view.render("columns", {"columns": columns, "file_path": file_path})

    def train(self, request: Request, view: View):
        file_path = request.input("file_path")
        x_col = request.input("x_col")
        y_col = request.input("y_col")
        title = request.input("title")
        x_label = request.input("x_label")
        y_label = request.input("y_label")

        df = pd.read_csv(file_path)
        X = df[[x_col]].values
        y = df[y_col].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
        
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        
        y_pred = regressor.predict(X_test)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Training set", "Test set"))
        
        fig.add_trace(
            go.Scatter(x=X_train.flatten(), y=y_train, mode='markers', name='Training Data', marker=dict(color='red')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=X_train.flatten(), y=regressor.predict(X_train), mode='lines', name='Regression Line', marker=dict(color='blue')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=X_test.flatten(), y=y_test, mode='markers', name='Test Data', marker=dict(color='red')),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=X_train.flatten(), y=regressor.predict(X_train), mode='lines', name='Regression Line', marker=dict(color='blue')),
            row=1, col=2
        )

        fig.update_layout(title_text=title, xaxis_title=x_label, yaxis_title=y_label, xaxis2_title=x_label, yaxis2_title=y_label)

        graph = fig.to_html(full_html=False)

        return view.render("results", {"graph": graph})