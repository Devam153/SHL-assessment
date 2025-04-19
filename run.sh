
#!/bin/bash

# Start Streamlit
streamlit run app.py

# Ensure the process terminates when the script exits
trap "kill 0" EXIT
