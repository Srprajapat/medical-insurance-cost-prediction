# Medical Insurance Cost Prediction

This project predicts medical insurance costs based on user input such as age, BMI, smoking status, and region. The model is built using machine learning techniques and deployed using FastAPI on Render.

## Features
- User-friendly web interface for input submission
- FastAPI-based backend for prediction
- Machine learning model trained on the Medical Cost Personal Dataset
- Deployment on Render for easy access

## Tech Stack
- **Programming Language:** Python  
- **Backend:** FastAPI  
- **Frontend:** HTML, CSS, JavaScript  
- **Machine Learning:** scikit-learn, XGBoost  
- **Deployment:** Render  

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Srprajapat/medical-insurance-cost-prediction.git
   cd medical-insurance-cost-prediction
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Update the HTML file (if running locally):
   Since the project is deployed on Render, some JavaScript code in `index.html` may reference the deployed URL. If you are running it locally, update the JavaScript part where the API is called:
   ```javascript
   fetch("https://micpbysr.onrender.com/predict", {
   ```
   Change it to:
   ```javascript
   fetch("http://127.0.0.1:8000/predict", {
   ```
5. Run the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
6. Open your browser and visit:
   ```
   http://127.0.0.1:8000
   ```

## Deployment on Render
- The application is deployed on Render and can be accessed at:
  ```
  https://micpbysr.onrender.com
  ```

## API Endpoints
### 1. Home Page
- **Endpoint:** `/`
- **Method:** GET
- **Description:** Serves the HTML form to collect user input.

### 2. Predict Insurance Cost
- **Endpoint:** `/predict`
- **Method:** POST
- **Payload:**
  ```json
  {
    "age": 30,
    "sex": 1,
    "bmi": 25.5,
    "children": 2,
    "smoker": 0,
    "region": 1
  }
  ```
- **Response:**
  ```json
  {
    "predicted_insurance_cost": 6793.14
  }
  ```

## Contributing

Feel free to fork this repository and make improvements. Pull requests are welcome! 🚀


## Author
Seetaram Prajapat - [GitHub Profile](https://github.com/Srprajapat)

## Contact

For any questions or suggestions, reach out to me at [**seetaram.22jics083@jietjodhpur.ac.in**](mailto\:seetaram.22jics083@jietjodhpur.ac.in) or connect on [LinkedIn](https://www.linkedin.com/in/seetaram-prajapat).

