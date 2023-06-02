from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import uvicorn

app = FastAPI()


@app.get('/')
def index():
    return {'message': 'Hello, World'}


# Define the input data model
class CropData(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: int


# Load the trained decision tree model
with open('DecisionTreeModel.pkl', 'rb') as file:
    model = pickle.load(file)


# Define the API endpoint for GET request
@app.get('/predict_crop')
def get_predict_crop():
    return {'message': 'Send a POST request to this endpoint with crop data.'}


# Define the API endpoint for POST request
@app.post('/predict_crop')
def post_predict_crop(request: Request, data: CropData):
    # Preprocess the input data if necessary
    input_data = [[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph
    ]]

    # Make predictions using the loaded model
    predicted_crop = model.predict(input_data)[0]

    # Return the predicted crop as the API response
    return {'crop': predicted_crop}

