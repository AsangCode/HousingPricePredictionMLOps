from flask import Flask, request, render_template
from prediction_pipeline import PredictPipeline, CustomData

app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    else:
        # Retrieve form data and convert to appropriate types
        data = CustomData(
            area=float(request.form.get("area")),
            bedrooms=float(request.form.get("bedrooms")),
            bathrooms=float(request.form.get("bathrooms")),
            stories=float(request.form.get("stories")),
            mainroad=request.form.get("mainroad"),
            guestroom=request.form.get("guestroom"),
            basement=request.form.get("basement"),
            hotwaterheating=request.form.get("hotwaterheating"),
            airconditioning=request.form.get("airconditioning"),
            parking=float(request.form.get("parking")),
            prefarea=request.form.get("prefarea"),
            furnishingstatus=request.form.get("furnishingstatus")
        )
        final_data=data.get_data_as_dataframe()

        predict_pipeline=PredictPipeline()

        pred=predict_pipeline.predict(final_data)

        result=round(pred[0],2)

        return render_template("result.html",final_result=result)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)


        
        