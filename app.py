"""
Step 9: Crime Prediction & Hotspot Detection - Streamlit Web App
User sign-in and admin approval; then upload data, view EDA, heatmap, and predictions.
"""
from pathlib import Path

import base64
import pandas as pd
import streamlit as st

# Must be first Streamlit command
st.set_page_config(page_title="Crime Prediction System", page_icon="🚔", layout="wide")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from auth import init_db, register, login, get_pending_users, set_user_status, reset_password
from data_preprocessing import preprocess, normalize_columns
from config import REQUIRED_COLUMNS
from analysis import plot_crime_types, plot_peak_hours, plot_monthly_trend
from heatmap import build_heatmap
from classification import (
    train_classification_model,
    train_risk_model,
    MODEL_DIR as CLASS_MODEL_DIR,
)
try:
    from forecast import prepare_trend_data, train_forecast_model, HAS_PROPHET
except ImportError:
    HAS_PROPHET = False
    prepare_trend_data = None
    train_forecast_model = None

init_db()

CLEANED_PATH = "crime_data_cleaned.csv"
MODELS_DIR = Path("models")


def _auth_css():
    st.markdown("""
    <style>
    /* Hide Streamlit default menu / deploy controls */
    #MainMenu {visibility: hidden;}
    header[data-testid="stHeader"] {display: none !important;}
    footer {visibility: hidden;}
    div[data-testid="stToolbar"] {display: none !important;}

    /* Simple, professional auth page */
    .stApp { background: #f5f6f8; }
    .block-container { padding-top: 5rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.25rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)


def _render_signin_form():
    """Render Sign In form; handle submit and return True if logged in."""
    msg_placeholder = st.empty()
    with st.form("login_form"):
        st.markdown("**Sign In**")
        st.text_input("Username", key="login_username", placeholder="Enter your username", label_visibility="collapsed")
        st.text_input("Password", type="password", key="login_password", placeholder="Enter your password", label_visibility="collapsed")
        st.markdown("")
        submitted = st.form_submit_button("Sign In", type="primary")
    if submitted:
        un = st.session_state.get("login_username", "")
        pw = st.session_state.get("login_password", "")
        ok, msg, user = login(un or "", pw or "")
        if ok:
            st.session_state["user"] = user
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            msg_placeholder.error(msg)


def _render_user_login_section():
    """Render user Sign In / Forgot Password toggle in one place."""
    if "login_view" not in st.session_state:
        st.session_state["login_view"] = "signin"

    # Forgot password view
    if st.session_state["login_view"] == "forgot":
        msg_placeholder = st.empty()
        with st.form("forgot_password_form"):
            st.markdown("**Reset password**")
            fp_user = st.text_input(
                "Username",
                placeholder="Enter your username",
                label_visibility="collapsed",
                key="fp_username",
            )
            fp_phone = st.text_input(
                "Phone Number",
                placeholder="Registered phone number",
                label_visibility="collapsed",
                key="fp_phone",
            )
            fp_new = st.text_input(
                "New Password",
                type="password",
                placeholder="New password (min 6 characters)",
                label_visibility="collapsed",
                key="fp_new",
            )
            fp_confirm = st.text_input(
                "Confirm New Password",
                type="password",
                placeholder="Confirm new password",
                label_visibility="collapsed",
                key="fp_confirm",
            )
            col_fp1, col_fp2 = st.columns(2)
            with col_fp1:
                submitted_fp = st.form_submit_button("Change password", type="primary", use_container_width=True)
            with col_fp2:
                back_fp = st.form_submit_button("Back to sign in", type="secondary", use_container_width=True)

        if back_fp:
            st.session_state["login_view"] = "signin"
            st.rerun()

        if submitted_fp:
            if not fp_user or not fp_phone or not fp_new or not fp_confirm:
                msg_placeholder.error("All fields are required.")
            elif fp_new != fp_confirm:
                msg_placeholder.error("Password and Confirm Password do not match.")
            else:
                ok, msg = reset_password(fp_user, fp_phone, fp_new)
                if ok:
                    msg_placeholder.success(msg)
                    st.session_state["login_view"] = "signin"
                    st.rerun()
                else:
                    msg_placeholder.error(msg)
        return

    # Sign in view (default)
    msg_placeholder = st.empty()
    with st.form("login_form_user"):
        st.markdown("**Sign In**")
        st.text_input(
            "Username",
            key="login_username",
            placeholder="Enter your username",
            label_visibility="collapsed",
        )
        st.text_input(
            "Password",
            type="password",
            key="login_password",
            placeholder="Enter your password",
            label_visibility="collapsed",
        )
        st.markdown("")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            submitted_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)
        with col_l2:
            forgot_clicked = st.form_submit_button("Forgot password", type="secondary", use_container_width=True)

    if forgot_clicked:
        st.session_state["login_view"] = "forgot"
        st.rerun()

    if submitted_login:
        un = st.session_state.get("login_username", "")
        pw = st.session_state.get("login_password", "")
        ok, msg, user = login(un or "", pw or "")
        if ok:
            st.session_state["user"] = user
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            msg_placeholder.error(msg)


def render_login_register():
    """Show User/Admin choice, then Sign In (and Register for User only)."""
    # If already logged in, do not render any auth UI again.
    if st.session_state.get("logged_in"):
        return

    _auth_css()

    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = None

    # --- Top header (fixed at top) ---
    st.markdown(
        """
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 999;
            background: linear-gradient(90deg, #1a1f2e 0%, #252b3b 100%);
            padding: 1rem 1.5rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">🚔</span>
                <div style="font-size: 1.25rem; font-weight: 600; color: #fff;">Crime Prediction System</div>
            </div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.75); margin-left: auto;">Home — Sign in or register to continue</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Hero: simple, professional header ---
    hero_image_path = Path(
        r"C:\Users\Sk Ayesha\.cursor\projects\c-Users-Sk-Ayesha-OneDrive-Documents-crime-prediction\assets\c__Users_Sk_Ayesha_AppData_Roaming_Cursor_User_workspaceStorage_c8ce522c2dd23c3aa1ad1223b0309c7e_images_image-376be812-187c-4248-807b-bb0d23934937.png"
    )
    if not hero_image_path.exists():
        hero_image_path = Path("assets/hero_banner.jpg")

    if hero_image_path.exists():
        with open(hero_image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        bg_css = f"linear-gradient(to bottom, rgba(20,25,35,0.5), rgba(20,25,35,0.75)), url('data:image/png;base64,{encoded}') center/cover no-repeat"
    else:
        bg_css = "linear-gradient(135deg, #1a1f2e, #252b3b)"

    st.markdown(
        f"""
        <div style="
            width: 100%;
            height: 320px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            background: {bg_css};
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            box-sizing: border-box;
            margin-bottom: 2rem;
        ">
            <div style="max-width: 640px; text-align: center;">
                <div style="font-size: 1.75rem; font-weight: 600; letter-spacing: 0.02em; color: #fff;">
                    Crime Prediction &amp; Hotspot Detection
                </div>
                <div style="font-size: 0.9rem; margin-top: 0.4rem; color: rgba(255,255,255,0.88);">
                    Analyze data, view hotspots, and forecast trends.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centered auth card
    col_side, col_center, col_side2 = st.columns([1, 1.5, 1])
    with col_center:
        st.markdown(
            '<p style="font-size: 0.85rem; color: #5f6368; margin-bottom: 0.75rem;">Sign in to continue</p>',
            unsafe_allow_html=True,
        )
        btn_user, btn_admin = st.columns(2)
        with btn_user:
            if st.button("User", type="primary" if st.session_state["auth_mode"] == "user" else "secondary", use_container_width=True):
                st.session_state["auth_mode"] = "user"
                st.rerun()
        with btn_admin:
            if st.button("Admin", type="primary" if st.session_state["auth_mode"] == "admin" else "secondary", use_container_width=True):
                st.session_state["auth_mode"] = "admin"
                st.rerun()
        st.markdown("---")

        if st.session_state["auth_mode"] is None:
            _render_signin_form()
            return False

        if st.session_state["auth_mode"] == "admin":
            _render_signin_form()
            return False

        tab_login, tab_register = st.tabs(["Sign In", "Register"])
        with tab_login:
            _render_user_login_section()
        with tab_register:
            reg_msg_placeholder = st.empty()
            with st.form("register_form"):
                st.markdown("**Create account**")
                st.text_input("Username", placeholder="Username (min 3 characters)", label_visibility="collapsed", key="reg_username")
                st.text_input("Password", type="password", placeholder="Password (min 6 characters)", label_visibility="collapsed", key="reg_password")
                st.text_input("Confirm Password", type="password", placeholder="Confirm password", label_visibility="collapsed", key="reg_confirm")
                st.text_input("Phone Number", placeholder="Phone number", label_visibility="collapsed", key="reg_phone")
                st.caption("Await admin approval after registration.")
                st.markdown("")
                submitted_reg = st.form_submit_button("Register", type="primary")
            if submitted_reg:
                run = st.session_state.get("reg_username", "")
                rpw = st.session_state.get("reg_password", "")
                rpw2 = st.session_state.get("reg_confirm", "")
                rphone = st.session_state.get("reg_phone", "")
                if not run or not rpw or not rpw2 or not rphone:
                    reg_msg_placeholder.error("All fields are required.")
                elif rpw != rpw2:
                    reg_msg_placeholder.error("Passwords do not match.")
                else:
                    ok, msg = register(run, rpw, rphone)
                    if ok:
                        reg_msg_placeholder.success(msg)
                    else:
                        reg_msg_placeholder.error(msg)
    return False


def ensure_models_dir():
    CLASS_MODEL_DIR.mkdir(exist_ok=True)


def get_cleaned_df():
    """Load cleaned data from disk if available."""
    p = Path(CLEANED_PATH)
    if p.exists():
        return pd.read_csv(p)
    return None


def render_heatmap_html(data: pd.DataFrame, height: int = 600):
    """Build Folium heatmap and return HTML for Streamlit."""
    m = build_heatmap(data, output_path=None)
    return m.get_root().render()


def main():
    if not st.session_state.get("logged_in"):
        render_login_register()
        return

    user = st.session_state.get("user", {})
    st.sidebar.write(f"Logged in as **{user.get('username', '')}**")
    if user.get("is_admin"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("👤 Admin")
        st.sidebar.caption("Approve or reject new user registrations below.")
        pending = get_pending_users()
        if pending:
            for u in pending:
                st.sidebar.write(f"**{u['username']}** — {u['phone']}")
                col_a, col_b = st.sidebar.columns(2)
                with col_a:
                    if st.button(f"Approve {u['username']}", key=f"approve_{u['id']}"):
                        set_user_status(u["id"], "approved")
                        st.rerun()
                with col_b:
                    if st.button(f"Reject {u['username']}", key=f"reject_{u['id']}"):
                        set_user_status(u["id"], "rejected")
                        st.rerun()
        else:
            st.sidebar.caption("No pending registrations.")
        st.sidebar.divider()
    if st.sidebar.button("Logout"):
        del st.session_state["logged_in"]
        del st.session_state["user"]
        st.rerun()

    st.title("🚔 Crime Prediction & Hotspot Detection System")
    st.caption("Analyze crime data, view hotspots, and forecast trends.")

    # Sidebar: data source
    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("Upload crime CSV", type=["csv"])
    use_cleaned = st.sidebar.checkbox("Use saved cleaned data", value=Path(CLEANED_PATH).exists())

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        raw = normalize_columns(raw)
        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            st.sidebar.error(f"Dataset must have columns (or mappings in config): {REQUIRED_COLUMNS}. Missing: {missing}")
            st.sidebar.info("Chicago data: use 'Primary Type', 'Date', 'Latitude', 'Longitude'.")
        else:
            with st.spinner("Preprocessing..."):
                data = preprocess(raw)
            data.to_csv(CLEANED_PATH, index=False)
            st.sidebar.success(f"Preprocessed {len(data):,} rows. Saved to {CLEANED_PATH}.")
            use_cleaned = True

    data = None
    if use_cleaned:
        data = get_cleaned_df()
    if data is None:
        st.info("Upload a crime CSV above or place `crime_data_cleaned.csv` in the project folder to start.")
        st.markdown("**Expected columns:** Crime Type, Date, Latitude, Longitude (or Chicago: Primary Type, Date, Latitude, Longitude)")
        return

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Data", "📊 EDA", "🗺️ Heatmap", "🔮 Predictions", "📈 Forecast"])

    with tab1:
        st.subheader("Crime data preview")
        st.dataframe(data.head(500), use_container_width=True)
        st.metric("Total records", len(data))

    with tab2:
        st.subheader("Crime pattern analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = plot_crime_types(data, top_n=15)
            st.pyplot(fig1)
            plt.close(fig1)
        with col2:
            fig2 = plot_peak_hours(data)
            st.pyplot(fig2)
            plt.close(fig2)
        fig3 = plot_monthly_trend(data)
        st.pyplot(fig3)
        plt.close(fig3)

    with tab3:
        st.subheader("Crime heatmap")
        try:
            html = render_heatmap_html(data, height=600)
            st.components.v1.html(html, height=600)
        except Exception as e:
            st.error(f"Could not build heatmap: {e}")

    with tab4:
        st.subheader("High-risk area & crime type prediction")
        ensure_models_dir()
        lgbm_path = MODELS_DIR / "lgbm_classifier.joblib"
        rf_path = MODELS_DIR / "rf_risk.joblib"

        if not lgbm_path.exists() or not rf_path.exists():
            if st.button("Train models (LightGBM + Random Forest)"):
                with st.spinner("Training..."):
                    try:
                        _, _, acc1 = train_classification_model(data)
                        _, _, acc2 = train_risk_model(data)
                        st.success(f"LightGBM accuracy: {acc1:.2%} | Random Forest accuracy: {acc2:.2%}")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            else:
                st.info("Click **Train models** to train classification and risk models.")
        else:
            import joblib
            lgbm = joblib.load(lgbm_path)
            le = joblib.load(MODELS_DIR / "label_encoder.joblib")
            feats = joblib.load(MODELS_DIR / "classification_features.joblib")
            st.write("**Predict crime type** (by time)")
            c1, c2, c3, c4 = st.columns(4)
            hour = c1.number_input("Hour", 0, 23, 12)
            month = c2.number_input("Month", 1, 12, 6)
            day = c3.number_input("Day", 1, 31, 15)
            day_of_week = c4.number_input("Day of week (0=Mon)", 0, 6, 2)
            row = {f: (hour if f == "hour" else month if f == "month" else day if f == "day" else day_of_week if f == "day_of_week" else 0) for f in feats}
            X = pd.DataFrame([row])[feats]
            pred = lgbm.predict(X)[0]
            label = le.inverse_transform([pred])[0]
            st.success(f"Predicted crime type: **{label}**")

    with tab5:
        st.subheader("Crime trend forecast (Prophet)")
        if not HAS_PROPHET:
            st.warning("Prophet is not installed. The **Forecast** tab needs it to predict future crime trends.")
            st.code("pip install prophet", language="text")
            st.caption("After installing, restart the app (stop with Ctrl+C, then run: python -m streamlit run app.py)")
        else:
            prophet_path = MODELS_DIR / "prophet_forecast.joblib"
            if not prophet_path.exists():
                if st.button("Train forecast model (Prophet)"):
                    with st.spinner("Fitting Prophet..."):
                        try:
                            trend = prepare_trend_data(data)
                            forecast, _ = train_forecast_model(trend, periods=30)
                            st.session_state["forecast_df"] = forecast
                            st.success("Forecast model trained.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                else:
                    st.info("Click **Train forecast model** to generate trend forecast.")
            else:
                import joblib
                periods = st.slider("Forecast days", 7, 90, 30)
                try:
                    trend = prepare_trend_data(data)
                    model = joblib.load(prophet_path)
                    future = model.make_future_dataframe(periods=periods)
                    forecast = model.predict(future)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast")
                    ax.fill_between(
                        forecast["ds"],
                        forecast["yhat_lower"],
                        forecast["yhat_upper"],
                        alpha=0.3,
                    )
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Crime count")
                    ax.set_title("Crime trend forecast")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(str(e))

    st.sidebar.divider()
   

if __name__ == "__main__":
    main()
