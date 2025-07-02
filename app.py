from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from prophet import Prophet
from fastapi.responses import PlainTextResponse
import json

app = FastAPI()

sales_df = pd.read_csv("medicine_sales_sample.csv")
sales_df.columns = sales_df.columns.str.strip().str.lower().str.replace(" ", "_")

class MedicineRequest(BaseModel):
    medicine_name: str
    months: int = 3

def predict_total_demand(medicine_name: str, months: int = 3) -> int:
    df_m = sales_df[sales_df["medicine_name"].str.lower() == medicine_name.lower()].copy()
    if df_m.empty:
        raise ValueError("Medicine not found")

    df_m["date"] = pd.to_datetime(df_m["date"])
    df_m = df_m.rename(columns={"date": "ds", "quantity_sold": "y"})

    model = Prophet()
    model.fit(df_m)

    future = model.make_future_dataframe(periods=30 * months)
    forecast = model.predict(future)

    forecast["month"] = forecast["ds"].dt.to_period("M")
    monthly_forecast = forecast.groupby("month")["yhat"].sum().reset_index()
    monthly_forecast["yhat"] = monthly_forecast["yhat"].apply(lambda x: max(0, int(x)))

    total_predicted = monthly_forecast.tail(months)["yhat"].sum()
    return int(total_predicted)
# ek k liye 
@app.post("/monthly-plan")
def post_monthly_stock_plan(req: MedicineRequest):
    try:
        total = predict_total_demand(req.medicine_name ,months=req.months )
        return {
            req.medicine_name: {
                "total_predicted_demand": total,
                  # Optional: We can't calculate percent without comparing to others
                #   "percentage_of_total": " This is for next Month"
            }
        }
    except ValueError:
        raise HTTPException(status_code=404, detail="Medicine not found")


@app.get("/monthly-plan", response_class=PlainTextResponse)
def get_monthly_stock_plan():
    total_demand_all = {}
    grand_total = 0

    for med in sales_df["medicine_name"].unique():
        try:
            total = predict_total_demand(med, months=3)
            total_demand_all[med] = total
            grand_total += total
        except:
            continue

    final_output = {}
    for med, demand in total_demand_all.items():
        percent = round((demand / grand_total) * 100, 2) if grand_total > 0 else 0.0
        final_output[med] = {
            "total_predicted_demand": demand,
            "percentage_of_total": percent
        }

    return json.dumps(final_output, indent=4)
