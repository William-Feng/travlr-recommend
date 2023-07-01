from typing import Union

from fastapi import FastAPI
from models.user_recommendation import run_user_recommendation_system
from models.location_recommendations import run_location_recommendation_system

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/users/{id}")
def get_user_preferences(id):
    return run_user_recommendation_system(id)
    
@app.get("/locations/{user_id}")
def get_location_preferences(user_id):
    return run_location_recommendation_system(user_id)