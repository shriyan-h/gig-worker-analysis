# 💸 Gig Worker Income Predictor

A machine learning web app that predicts estimated net income for gig economy workers based on shift details, platform, location, and performance metrics. Built with Streamlit and scikit-learn.

---

## 🚀 Live Demo
https://gig-worker-analysis-gl6memnnhnxaf7rjitqdsw.streamlit.app/

---

## 📸 Preview

> _Add a screenshot of the app here_

---

## 🧠 What It Does

- Takes shift inputs like platform, city, weather, hours worked, orders completed, and distance
- Passes them through a trained ML model (Random Forest)
- Outputs a predicted net income in INR
- Dropdown options are dynamically pulled from the real dataset

---

## 🗂️ Project Structure

```
gig-worker/
│
├── app.py                        # Streamlit frontend
├── GiG.ipynb                     # Model training notebook
├── gig_worker_income_clean.csv   # Cleaned dataset
├── gig_worker_model.pkl          # Trained model
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| ML Model | scikit-learn (Random Forest) |
| Data | pandas, numpy |
| Model Persistence | joblib / pickle |
| Language | Python 3.11 |

---

## 📦 Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/gig-worker-income-predictor.git
cd gig-worker-income-predictor
```

**2. Install dependencies**
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly
```

**3. Run the app**
```bash
python -m streamlit run app.py
```

---

## 🎯 Input Features

| Feature | Type | Description |
|---|---|---|
| Platform | Categorical | Swiggy, Zomato, Uber, etc. |
| City | Categorical | Worker's city |
| Weather | Categorical | Weather during shift |
| Worker Type | Categorical | Full-time, part-time, etc. |
| Vehicle Type | Categorical | Bike, car, cycle, etc. |
| Age | Numeric | Worker age |
| Experience | Numeric | Years of experience |
| Hours Worked | Numeric | Shift duration |
| Orders Completed | Numeric | Total deliveries |
| Distance (km) | Numeric | Total km covered |
| Customer Rating | Numeric | Shift rating (1–5) |
| Weekend | Boolean | Weekday vs weekend |

---

## 📊 Model

- **Algorithm:** Random Forest Regressor
- **Training:** `GiG.ipynb`
- **Saved as:** `gig_worker_model.pkl`
- **Target variable:** Net income (INR)

---

## 👥 Team

- **Shriyansh Soni** — [github.com/shriyan-h](https://github.com/shriyan-h)
- **Anishka** — [github.com/AnishkaKaithwas](https://github.com/AnishkaKaithwas)
- **Manya** — [github.com/manyag692-create](https://github.com/manyag692-create)

---

## 📄 License

This project is for academic and portfolio purposes.