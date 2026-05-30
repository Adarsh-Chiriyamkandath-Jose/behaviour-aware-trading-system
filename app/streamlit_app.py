"""Streamlit dashboard for the Behaviour-Aware Trading System.

Run with:  streamlit run app/streamlit_app.py
"""
import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Behaviour-Aware Trading System", layout="wide")
    st.title("Behaviour-Aware Trading System")
    st.write("Dashboard scaffold — TODO: ticker selector, predictions, equity curve, SHAP.")


if __name__ == "__main__":
    main()
