# import requests for interacting with backend
import requests

# import streamlit library for IO
import streamlit as st

# import pandas
import pandas as pd

# define functiom which can provide formatted input with appropriate label and input text
# this will help in p[roducing consistent representation
def formatted_number_input(title, hint, **kwargs):
    st.markdown(f"**{title}**")
    st.caption(hint)
    return st.number_input("", **kwargs)

def formatted_number_input2(title, hint, minval, maxval, defvalue, steps, valformat="%.6f"):

    st.markdown('<div style="margin-bottom:4px;">', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1], vertical_alignment="center")

    with col1:
        st.markdown(
            f"""
            <div style="line-height:1.0">
                <strong>{title}</strong><br>
                <span style="font-size:1.10em; color:gray;">(Range {hint})</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        usre_input = st.number_input("",
                            min_value=minval,
                            max_value=maxval,
                            value=defvalue,
                            step=steps,
                            format=valformat,
                            label_visibility="collapsed"
                            )

    st.markdown('</div>', unsafe_allow_html=True)

    return usre_input

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenenace App"  #,
    #layout="wide"
)


st.markdown("""
<style>
.block-container {
    padding-top: 0.75rem;
    padding-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🏖️ Predict Maintenance")
st.write("The Predict Maintenance app is a tool to predict if an Engine needs any maintenance based on provided operating sensor parameters.")
st.write("Fill in the details below and click **Predict** to see if the Engine needs maintenance to prevent from failure.")



# ====================================
# Section : Capture Engine Parameters
# ====================================
st.subheader ("Engine Parameters")

rpm = formatted_number_input2(
    "Lubricating oil pressure in kilopascals (kPa)",
    "50 to 2500",
    minval=50.0,
    maxval=2500.0,
    defvalue=735.0,
    steps=10.0,
    valformat="%.2f"
)


oil_pressure = formatted_number_input2(
    "Lubricating oil pressure in kilopascals (kPa)",
    "0.001 to 10.0",
    minval=0.001,
    maxval=10.0,
    defvalue=3.300000,
    steps=0.001,
    valformat="%.6f"
)


fuel_pressure = formatted_number_input2(
    "Fuel Pressure in kilopascals (kPa)",
    "0.01 to 25.0",
    minval=0.01,
    maxval=25.0,
    defvalue=6.500000,
    steps=0.01,
    valformat="%.6f"
)


coolant_pressure = formatted_number_input2(
    "Coolant Pressure in kilopascals (kPa)",
    "0.01 to 10.0",
    minval=0.01,
    maxval=10.0,
    defvalue=2.250000,
    steps=0.10,
    valformat="%.6f"
)


lub_oil_temp = formatted_number_input2(
    "Lubricating oil Temperature in degrees Celsius (°C)",
    "50.0 to 100.0",
    minval=50.0,
    maxval=100.0,
    defvalue=75.0,
    steps=0.1,
    valformat="%.6f"
)


coolant_temp = formatted_number_input2(
    "Coolant Temperature in degrees Celsius (°C)",
    "50.0 to 200.0",
    minval=50.0,
    maxval=200.0,
    defvalue=75.000000,
    steps=0.1,
    valformat="%.6f"
)


# ==========================
# Single Value Prediction
# ==========================
if st.button("Check fo Maintenance"):

    # extract the data collected into a structure
    input_data = {
        'Engine_rpm'              : float(rpm),
        'Lub_oil_pressure'        : float(oil_pressure),
        'Fuel_pressure'           : float(fuel_pressure),
        'Coolant_pressure'        : float(coolant_pressure),
        'lub_oil_temp'            : float(lub_oil_temp),
        'Coolant_temp'            : float(coolant_temp),
    }

    input_df = pd.DataFrame([input_data])

    response = requests.post (
        "https://harishsohani-AIMLPredictiveMaintenanceBackEnd.hf.space/v1/EngPredMaintenance",
        json=input_data
        )

    if response.status_code == 200:
        ## get result as json
        result = response.json ()

        resp_status = result.get ("status")

        if resp_status == "success":

            ## Get Sales Prediction Value
            prediction_from_backend = result.get ("prediction")  # Extract only the value

            # generate output string
            if prediction_from_backend == 1:
                resultstr = "Engine **likely** needs maintenance."
            else:
                resultstr = "Engine does not need any maintenance"

            st.success(resultstr)

        else:

            error_str = result.get ("message")

            st.error(error_str)

    elif response.status_code == 400 or response.status_code == 500:  # known errors

        ## get result as json
        result = response.json ()

        error_str = result.get ("message")
        st.error (f"Error processing request- Status Code : {response.status_code}, error : {error_str}")

    else:
        st.error (f"Error processing request- Status Code : {response.status_code}")

    # Show the etails of data frame prepared from user input
    st.subheader("📦 Input Data Summary")
    st.dataframe (input_df)
