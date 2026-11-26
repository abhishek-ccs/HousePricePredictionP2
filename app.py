
import html
from pathlib import Path

import joblib
import streamlit as st

from model_utils import CSV_PATH, match_city, parse_locations, predict_price, train


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="HouseWise AI | Property Valuation",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# STYLING: DARK NAVY, SUBTLE GREEN, NO PURPLE GRADIENT
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --background: #101722;
        --panel: #17212e;
        --input: #111b28;
        --border: #2b394a;
        --text: #f1f5f9;
        --muted: #a5b1c1;
        --accent: #80d8b6;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: var(--background);
    }

    [data-testid="stSidebar"] {
        background: #131d2a;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: clamp(2rem, 4vw, 3.4rem) !important;
        font-weight: 800 !important;
        line-height: 1.15 !important;
    }

    .hero {
        background: #172331;
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .hero-description {
        color: #bac6d5;
        font-size: 1.05rem;
        max-width: 760px;
        line-height: 1.8;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1.1rem;
    }

    .panel-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .panel-subtitle {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 1.2rem;
    }

    /* Inputs: consistent appearance */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background: var(--input) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        min-height: 3.1rem;
        font-size: 1rem !important;
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #e0e7ef !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: var(--accent) !important;
        color: #10251e !important;
        border: none !important;
        border-radius: 10px !important;
        min-height: 3.2rem;
        font-weight: 800 !important;
        transition: filter 0.2s ease;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        filter: brightness(1.08);
    }

    /* Prediction result */
    .price-card {
        background: #192a30;
        border: 1px solid #426b60;
        border-radius: 18px;
        padding: 1.7rem;
        margin: 1rem 0;
    }

    .price-label {
        color: #a6e5cf;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 0.78rem;
        font-weight: 800;
    }

    .price-value {
        color: #e7fff5;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.2rem, 4vw, 3.8rem);
        font-weight: 800;
        line-height: 1.3;
        overflow-wrap: anywhere;
        margin: 0.5rem 0;
    }

    .price-note {
        color: #bdcec9;
        font-size: 0.9rem;
    }

    /* Property summary */
    .summary-card {
        background: #151f2c;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.3rem;
        color: #dce5ef;
        line-height: 1.9;
        overflow-wrap: anywhere;
    }

    .summary-card strong {
        color: #ffffff;
    }

    .detail-card {
        background: #1b2736;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.1rem;
        margin-bottom: 0.7rem;
        min-height: 100px;
    }

    .detail-label {
        color: var(--muted);
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }

    .detail-value {
        color: var(--text);
        font-family: 'Manrope', sans-serif;
        font-size: 1.2rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .footer {
        text-align: center;
        color: #8190a0;
        font-size: 0.82rem;
        padding-top: 2rem;
    }

    hr {
        border-color: var(--border) !important;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 0.7rem 2rem;
        }

        .hero {
            padding: 1.2rem;
            border-radius: 15px;
        }

        .panel {
            padding: 1rem;
        }

        .price-card {
            padding: 1.2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD THE MODEL
# Keep house_model.pkl beside app.py. If the pickle cannot be loaded
# (for example after a scikit-learn version change on the server),
# the model is retrained automatically from the CSV file.
# --------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "house_model.pkl"


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        loaded = joblib.load(MODEL_PATH)
        _ = loaded.feature_names_in_  # make sure this is the sub-area model
        return loaded
    except Exception:
        if not Path(CSV_PATH).exists():
            return None
        retrained, _ = train(CSV_PATH)
        return retrained


regression = load_model()

if regression is None:
    st.error(
        "Model file not found. Keep house_model.pkl and "
        "house_price_dataset_with_floors.csv in the same folder as app.py."
    )
    st.stop()

# City -> list of its sub-areas, read straight from the model's features.
sub_areas_by_city, known_city_names = parse_locations(
    list(regression.feature_names_in_)
)
ANY_SUBAREA = "Any sub-area (city average)"


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🏡 HouseWise AI")
    st.caption("Smart Property Valuation")
    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠 Property Predictor")
    st.markdown("📊 Property Valuation Report")
    st.markdown("📍 City & Sub-area Estimates")

    st.divider()
    st.caption(
        "Powered by your existing trained "
        "machine learning model."
    )


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered property insights</div>
        <h1>Find Your Property's Value</h1>
        <div class="hero-description">
            Enter the details of the house you are searching for.
            Get a machine-learning price estimate and a detailed
            breakdown of your selected property.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# MAIN LAYOUT
# Inputs are vertical: ONE FEATURE PER LINE
# --------------------------------------------------
input_column, result_column = st.columns(
    [0.9, 1.1],
    gap="large",
)


with input_column:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">🏘️ Property Details</div>
            <div class="panel-subtitle">
                Enter each property feature below.
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Location widgets live OUTSIDE the form so the sub-area list
    # updates as soon as the city is typed (press Enter).
    city = st.text_input(
        "City",
        value="Delhi",
        placeholder="Type any city, e.g. Bangalore, Pune, Goa...",
        help="Type a city name and press Enter. Sub-areas appear for supported cities.",
    ).strip()

    resolved_city = match_city(city, known_city_names)

    if resolved_city:
        sub_options = [ANY_SUBAREA] + sub_areas_by_city.get(resolved_city, [])
        sub_choice = st.selectbox(
            f"Sub-area in {resolved_city}",
            sub_options,
            help=f"Only sub-areas of {resolved_city} are listed.",
        )
    elif city:
        sub_choice = st.selectbox(
            "Sub-area",
            [ANY_SUBAREA],
            help="Sub-area data is not available for this city yet.",
        )
        st.caption(
            f"No sub-area data for “{city}” yet, so “Any sub-area” is used. "
            "Supported cities: " + ", ".join(known_city_names) + "."
        )
    else:
        sub_choice = ANY_SUBAREA

    sub_area = None if sub_choice == ANY_SUBAREA else sub_choice

    with st.form("property_prediction_form"):

        # Each input occupies its own full-width row.
        area = st.number_input(
            "Area (sq. ft.)",
            min_value=100,
            value=1200,
            step=50,
        )

        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=1,
            value=2,
            step=1,
        )

        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=1,
            value=2,
            step=1,
        )

        parking = st.number_input(
            "Parking Spaces",
            min_value=0,
            value=1,
            step=1,
        )

        floors = st.number_input(
            "Number of Floors",
            min_value=1,
            value=1,
            step=1,
        )

        submitted = st.form_submit_button(
            "Predict Property Price",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">How It Works</div>
            <div class="panel-subtitle">
                Your existing trained regression model calculates
                the estimate. This interface does not retrain
                or replace your model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# VALUATION REPORT
# --------------------------------------------------
with result_column:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">📋 Property Valuation Report</div>
            <div class="panel-subtitle">
                Your predicted price and property details.
            </div>
        """,
        unsafe_allow_html=True,
    )

    if submitted:

        if not city:
            st.error("Please enter a city.")

        else:
            # ------------------------------------------
            # PREDICTION LOGIC
            # ------------------------------------------
            city_specific = resolved_city is not None

            prediction = predict_price(
                regression,
                area=area,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                parking=parking,
                floors=floors,
                city=resolved_city,
                sub_area=sub_area,
            )

            if city_specific and sub_area:
                prediction_type = "Predicted Property Price"
                prediction_note = (
                    f"This prediction uses the trained features for "
                    f"{sub_area}, {resolved_city}."
                )
            elif city_specific:
                prediction_type = "Predicted Property Price"
                prediction_note = (
                    f"No sub-area selected, so this is the average "
                    f"estimate across the sub-areas of {resolved_city}."
                )
            else:
                prediction_type = "Estimated Property Price"
                prediction_note = (
                    f"{city} was not in the model's training "
                    "cities, so no sub-area option is available. "
                    "This result is an average across the model's "
                    "known cities, not a city-specific prediction."
                )

            # ------------------------------------------
            # LARGE PRICE DISPLAY
            # ------------------------------------------
            st.markdown(
                f"""
                <div class="price-card">
                    <div class="price-label">
                        {prediction_type}
                    </div>
                    <div class="price-value">
                        ₹{prediction:,.0f}
                    </div>
                    <div class="price-note">
                        Calculated using your existing ML model
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Escape user-entered text before displaying in HTML.
            safe_city = html.escape(city)
            safe_sub_area = html.escape(sub_area) if sub_area else ""
            safe_location = (
                f"{safe_sub_area}, {safe_city}" if sub_area else safe_city
            )

            st.markdown(
                "### Property You Are Searching For"
            )

            st.markdown(
                f"""
                <div class="summary-card">
                    The property you are searching for is located
                    in <strong>{safe_location}</strong>, has an area of
                    <strong>{area:,.0f} sq. ft.</strong>, contains
                    <strong>{bedrooms} bedroom(s)</strong> and
                    <strong>{bathrooms} bathroom(s)</strong>, and
                    has <strong>{parking} parking space(s)</strong>
                    and <strong>{floors} floor(s)</strong>.

                    The {prediction_type.lower()} is
                    <strong>₹{prediction:,.0f}</strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("")

            # ------------------------------------------
            # PROPERTY FEATURE CARDS
            # ------------------------------------------
            st.markdown("### Property Specifications")

            details = [
                ("📐", "Property Area", f"{area:,.0f} sq. ft."),
                ("🛏️", "Bedrooms", str(bedrooms)),
                ("🛁", "Bathrooms", str(bathrooms)),
                ("🚗", "Parking Spaces", str(parking)),
                ("🏢", "Number of Floors", str(floors)),
                ("📍", "City", safe_city),
                (
                    "🏘️",
                    "Sub-area",
                    safe_sub_area or "Not selected / not available",
                ),
                (
                    "💰",
                    "Estimated Price per sq. ft.",
                    f"₹{prediction / area:,.0f}",
                ),
            ]

            # Display specifications in two columns.
            # The INPUTS above remain one feature per line.
            for index in range(0, len(details), 2):
                card_columns = st.columns(2, gap="medium")

                for column, detail in zip(
                    card_columns,
                    details[index:index + 2],
                ):
                    icon, label, value = detail

                    with column:
                        st.markdown(
                            f"""
                            <div class="detail-card">
                                <div class="detail-label">
                                    {icon} &nbsp; {label}
                                </div>
                                <div class="detail-value">
                                    {value}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            if city_specific:
                st.success(prediction_note)
            else:
                st.info(prediction_note)

            st.caption(
                "This is a machine-learning estimate, not a "
                "verified market valuation or guaranteed sale price."
            )

    else:
        st.markdown(
            """
            <div class="price-card">
                <div class="price-label">
                    Your Estimated Property Price
                </div>
                <div class="price-value">₹ —</div>
                <div class="price-note">
                    Enter the property details and click
                    Predict Property Price to generate your report.
                </div>
            </div>

            <div class="summary-card">
                <strong>Your report will include:</strong><br>
                • A prominent predicted price<br>
                • A detailed property description<br>
                • Area, bedrooms, bathrooms, parking and floors<br>
                • The selected city and sub-area<br>
                • An explanation of city-specific or fallback estimates
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        HOUSEWISE AI · Property price estimates powered by
        your existing machine learning model
    </div>
    """,
    unsafe_allow_html=True,
)
