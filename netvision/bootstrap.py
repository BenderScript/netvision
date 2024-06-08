from streamlit.web import bootstrap

real_script = 'main_streamlit.py'
bootstrap.run(real_script,  False, ["run.py", f"{real_script}"], {})