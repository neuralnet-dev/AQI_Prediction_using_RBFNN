# AQI Prediction Using RBFNN

This project applies a **Radial Basis Function Neural Network (RBFNN)** to predict the **Air Quality Index (AQI)** based on pollutant data collected from Indian cities. The model is trained using the `city_day.csv` dataset, which contains daily air pollution readings along with AQI values.  
Built and developed by Amith S Patil, Asher Jarvis Pinto, Henry Gladson, Fariza Nuha Farooq, and Lavanya Devadiga.

---

## Dataset

- **Source**: [city_day.csv](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Features**: Pollutants such as PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Target**: `AQI` (Air Quality Index)

---

## Workflow

1. **Data Preprocessing**
   - Dropped rows with missing values in the `AQI` column
   - Dropped `Date` and `AQI_Bucket` columns
   - Filled remaining missing values using **median imputation**
   - Encoded the `City` column using **Label Encoding**

2. **Exploratory Data Analysis**
   - Visualized correlations between features using a **heatmap**

3. **Modeling with RBFNN**
   - Standardized features using **StandardScaler**
   - Used **KMeans clustering** to identify RBF centers (with optimal clusters chosen using the **elbow method**)
   - Computed **Radial Basis Function (Gaussian kernel)** using the median pairwise cluster distance
   - Calculated weights `W` using **pseudo-inverse regression**
   - Made predictions for both training and testing sets using the RBF network

4. **Evaluation**
   - Metrics:
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - R² Score
   - Visuals:
     - **Scatter plot** of actual vs. predicted AQI
     - **Line plot** comparing actual and predicted AQI over a subset of test samples


---

## Model Performance ( Non-Scaled )

```
MSE: 2442.1114451020526
RMSE: 49.417723997590706
R² Score: 0.8725485373085805
```

Make sure `city_day.csv` is in the same directory.