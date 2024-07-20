from masonite.routes import Route

ROUTES = [
    Route.get("/", "WelcomeController@show"),
    Route.get("/show", "SimpleLinearRegressionController@show").name("simplelinearregression.show"),
    Route.post("/upload", "SimpleLinearRegressionController@upload").name("simplelinearregression.upload"),
    Route.get("/dataset", "SimpleLinearRegressionController@dataset").name("simplelinearregression.dataset"),
    Route.post("/trainstored", "SimpleLinearRegressionController@trainstored").name("simplelinearregression.trainstored"),
    Route.post("/train", "SimpleLinearRegressionController@train").name("simplelinearregression.train"),
]
