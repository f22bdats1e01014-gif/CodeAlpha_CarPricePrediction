"""
CodeAlpha Task 3 — Car Price Prediction
Streamlit Web App (Aesthetic Edition, v2 — fixed empty-box bug)

Run with:
    streamlit run app.py

Requires outputs/car_price_model.pkl and outputs/preprocessing_artifacts.pkl
to exist (run the notebook once first).
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
    }

    /* Hero header */
    .hero {
        padding: 2.5rem 2rem 1.5rem 2rem;
        text-align: center;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C63FF 0%, #00D9C0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #9ca3af;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Native bordered containers -> styled as cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 0.4rem 0.4rem;
    }

    .card-heading {
        color: #e5e7eb;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0.2rem 0 1rem 0;
    }

    /* Prediction result card */
    .result-card {
        background: linear-gradient(135deg, #6C63FF 0%, #00D9C0 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
    }
    .result-card .label {
        color: rgba(255,255,255,0.85);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .result-card .price {
        color: white;
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .result-card .sub {
        color: rgba(255,255,255,0.85);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Metric chips row */
    .chip-row {
        display: flex;
        gap: 0.6rem;
        margin-top: 1.1rem;
        flex-wrap: wrap;
        justify-content: center;
    }
    .chip {
        background: rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 0.4rem 0.9rem;
        color: white;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 2.5rem 1rem;
        color: #6b7280;
    }
    .empty-state .icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6C63FF 0%, #00D9C0 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 14px rgba(108, 99, 255, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.45);
        color: white;
        border: none;
    }

    /* Section label pill */
    .pill {
        display: inline-block;
        background: rgba(108, 99, 255, 0.15);
        color: #a5b4fc;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        padding: 2rem 0 1rem 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# Hero
# ----------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🚗 Car Price Predictor</h1>
    <p>Instant resale value estimates powered by machine learning — trained on real used-car listings.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load("outputs/car_price_model.pkl")
    artifacts = joblib.load("outputs/preprocessing_artifacts.pkl")
    return model, artifacts


try:
    model, artifacts = load_artifacts()
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found. Please run the notebook (`Car_Price_Prediction.ipynb`) "
        "first to generate `outputs/car_price_model.pkl` and `outputs/preprocessing_artifacts.pkl`."
    )
    st.stop()

le_fuel = artifacts["fuel"]
le_seller = artifacts["seller"]
le_trans = artifacts["transmission"]
brand_freq = artifacts["brand_freq"]
feature_order = artifacts["feature_order"]
model_name = artifacts["model_name"]

# ----------------------------------------------------------------
# Layout: inputs (left) + result (right)
# ----------------------------------------------------------------
left, right = st.columns([1.3, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<span class="pill">Vehicle Details</span>', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">💰 Pricing & Mileage</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            present_price = st.number_input(
                "Present Price (Lakh ₹)", min_value=0.0, max_value=100.0, value=6.5, step=0.1,
                help="The car's current ex-showroom price if bought new today."
            )
            car_age = st.slider("Car Age (years)", min_value=0, max_value=25, value=5)
        with c2:
            driven_kms = st.number_input(
                "Kilometers Driven", min_value=0, max_value=500000, value=40000, step=500
            )
            owner = st.select_slider("Previous Owners", options=[0, 1, 3], value=0)

    st.write("")  # spacing between cards

    with st.container(border=True):
        st.markdown('<span class="pill">Specifications</span>', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">⚙️ Type & Model</div>', unsafe_allow_html=True)

        c3, c4, c5 = st.columns(3)
        with c3:
            fuel_type = st.selectbox("Fuel Type", options=list(le_fuel.classes_))
        with c4:
            selling_type = st.selectbox("Seller Type", options=list(le_seller.classes_))
        with c5:
            transmission = st.selectbox("Transmission", options=list(le_trans.classes_))

        brand_options = sorted(brand_freq.index.tolist())
        default_idx = brand_options.index("swift") if "swift" in brand_options else 0
        brand = st.selectbox(
            "Car Model / Brand token",
            options=brand_options,
            index=default_idx,
            help="First word of the car's model name, e.g. 'swift', 'ciaz', 'innova'."
        )

    st.write("")
    predict_clicked = st.button("✨ Predict Selling Price")

with right:
    with st.container(border=True):
        st.markdown(f'<span class="pill">Model: {model_name}</span>', unsafe_allow_html=True)

        if predict_clicked:
            brand_frequency = brand_freq.get(brand.lower(), brand_freq.mean())

            row = pd.DataFrame([{
                "Present_Price": present_price,
                "Driven_kms": driven_kms,
                "Fuel_Type": le_fuel.transform([fuel_type])[0],
                "Selling_type": le_seller.transform([selling_type])[0],
                "Transmission": le_trans.transform([transmission])[0],
                "Owner": owner,
                "Car_Age": car_age,
                "Brand_Freq": brand_frequency,
            }])[feature_order]

            prediction = model.predict(row)[0]
            depreciation_pct = (1 - prediction / present_price) * 100 if present_price > 0 else 0

            st.markdown(f"""
            <div class="result-card">
                <div class="label">Predicted Selling Price</div>
                <div class="price">₹{prediction:.2f}L</div>
                <div class="sub">≈ {depreciation_pct:.1f}% below present price</div>
                <div class="chip-row">
                    <div class="chip">🛣️ {driven_kms:,} km</div>
                    <div class="chip">📅 {car_age} yrs old</div>
                    <div class="chip">⛽ {fuel_type}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">🔮</div>
                <p>Fill in the car's details and click<br><b>Predict Selling Price</b> to see the estimate.</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Built for the CodeAlpha Data Science Internship — Task 3: Car Price Prediction with Machine Learning
</div>
""", unsafe_allow_html=True)