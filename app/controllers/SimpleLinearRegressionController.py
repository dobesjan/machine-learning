from masonite.request import Request
from masonite.filesystem import Storage
from masonite.views import View
from masonite.controllers import Controller
from masonite.validation import Validator
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def get_file_path(file_name: str):
    return os.path.join("storage", "uploads", file_name)

def load_dataset(file_path: str, storage: Storage):
    file_content = storage.disk('local').get(file_path)
    df = pd.read_csv(StringIO(file_content))
    return df.columns.tolist()

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
        file_path = get_file_path(dataset.filename)
        target_dir = os.path.join("storage", "uploads")
        file_name = os.path.splitext(dataset.filename)[0]
        storage.disk('local').put_file(target_dir, dataset, file_name)

        columns = load_dataset(file_path, storage)

        return view.render("columns", {"columns": columns, "file_path": file_path})
    
    def dataset(self, view: View):
        storage_dir = os.path.join("storage", "uploads")
        files = os.listdir(storage_dir)
        
        # Filter the list to include only files, excluding subdirectories
        files = [f for f in files if os.path.isfile(os.path.join(storage_dir, f))]
        return view.render("dataset", {"files": files})

    def trainstored(self, request: Request, view: View, storage: Storage):
        dataset = request.input("dataset")
        file_path = get_file_path(dataset.filename)
        columns = load_dataset(file_path, storage)
        return view.render("columns", {"columns": columns, "file_path": file_path})

    def train(self, request: Request, view: View, storage: Storage):
        file_name = request.input("file_name")
        file_path = os.path.join("storage", "uploads", file_name)
        target_dir = os.path.join("storage", "uploads")
        x_col = request.input("x_col")
        y_col = request.input("y_col")
        title = request.input("title")
        x_label = request.input("x_label")
        y_label = request.input("y_label")

        file_content = storage.disk('local').get(file_path)
        df = pd.read_csv(StringIO(file_content))
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