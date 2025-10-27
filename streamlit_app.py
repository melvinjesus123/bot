# Wrapper para Streamlit Cloud: ejecuta el verdadero app en scripts/streamlit_app.py
import runpy

if __name__ == "__main__":
    runpy.run_path("scripts/streamlit_app.py", run_name="__main__")
