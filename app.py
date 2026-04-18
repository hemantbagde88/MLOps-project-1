from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

# Project imports
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
from src.pipline.training_pipeline import TrainPipeline

# Initialize app
app = FastAPI()

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Routes
# =========================

# Home Page
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        name="vehicledata.html",
        context={"context": "Rendering"},
        request=request   # ✅ IMPORTANT FIX
    )


# Train Model
@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


# Prediction
@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = await request.form()

        data = {
            "Gender": int(form.get("Gender")),
            "Age": int(form.get("Age")),
            "Driving_License": int(form.get("Driving_License")),
            "Region_Code": float(form.get("Region_Code")),
            "Previously_Insured": int(form.get("Previously_Insured")),
            "Annual_Premium": float(form.get("Annual_Premium")),
            "Policy_Sales_Channel": float(form.get("Policy_Sales_Channel")),
            "Vintage": int(form.get("Vintage")),
            "Vehicle_Age_lt_1_Year": int(form.get("Vehicle_Age_lt_1_Year")),
            "Vehicle_Age_gt_2_Years": int(form.get("Vehicle_Age_gt_2_Years")),
            "Vehicle_Damage_Yes": int(form.get("Vehicle_Damage_Yes")),
        }

        # Convert to model input
        vehicle_data = VehicleData(**data)
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        # Prediction
        model_predictor = VehicleDataClassifier()
        value = model_predictor.predict(dataframe=vehicle_df)[0]

        status = "Response-Yes" if value == 1 else "Response-No"

        return templates.TemplateResponse(
            name="vehicledata.html",
            context={"context": status},
            request=request   # ✅ IMPORTANT FIX
        )

    except Exception as e:
        return {"status": False, "error": str(e)}


# Run Server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)