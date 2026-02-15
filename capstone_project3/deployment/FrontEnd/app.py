# import requests for interacting with backend
import requests

# import streamlit library for IO
import streamlit as st

# import pandas
import pandas as pd

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenenace App",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
#st.title("🏖️ Predict Engine Maintenance")
#st.write("The Predict Maintenance app is a tool to predict if an Engine needs any maintenance based on provided operating sensor parameters.")
#st.write("Fill in the details below and click **Predict** to see if the Engine needs maintenance to prevent from failure.")
# -----------------------------
# Title & Description
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 0.75rem;
    padding-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Set Page TITLE and additional information for consumer
# ---------------------------------------------------------
st.title("🏖️ Predict Maintenance")
st.markdown("""
The Predict Maintenance app help to predict if an engine needs maintenance based on operating sensor parameters.  
*Suggested ranges are based on known information - input is not restricted to the specified range* 
""")


# generic function to provide input 
# this is provided as an utiity to bring consistent user interface
# currently few parameters are used, rest or for later expansion
def formatted_number_input(title, hint, minval, maxval, defvalue, steps, valformat="%.4f"):

    st.markdown('<div style="margin-bottom:4px;">', unsafe_allow_html=True)
    
    user_input = st.number_input(
        label=f"{title}  ({hint})",
        #min_value=minval,
        #max_value=maxval,
        value=defvalue,
        #step=steps,
        format=valformat,
        #label_visibility="collapsed"
    )
         
    return user_input


st.markdown("""
<style>
/* Reduce top padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
/* Shared card styling */
.card {
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
    transition: 0.3s ease-in-out;
}
/* INPUT CARD */
.input-card {
    background: linear-gradient(145deg, #0f172a, #111827);
    border: 1px solid #334155;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.15);
}
/* OUTPUT CARD */
.output-card {
    background: linear-gradient(145deg, #111827, #0b1220);
    border: 1px solid #16a34a;
    box-shadow: 0 0 12px rgba(34,197,94,0.25);
}
/* Card title */
.card-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 16px;
    letter-spacing: 0.5px;
}
/* Button styling */
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #1e40af);
}
</style>
""", unsafe_allow_html=True)

# ====================================
# Section : Capture Engine Parameters
# ====================================
#st.subheader ("Engine Parameters")

# divide UI into two column layout by defining two columns 
# left column is used for input and right for output
col_inputs, col_output = st.columns([3, 1.5])

# update contnent (input) in left input column
with col_inputs:

    # using form for usre input, this proivides elegant interface
    with st.form("engine_form"):

        # set header string
        st.subheader("🔧 Engine Parameters")

        # create two columsn so we can spread the input into two columns
        col_left, col_right = st.columns(2)
    
        # define inputs in left column
        with col_left:
        
            rpm = formatted_number_input(
                "Engine RPM",
                "50 to 2500",
                minval=50.0,
                maxval=2500.0,
                defvalue=735.0,
                steps=10.0,
                valformat="%.2f"
            )
            
            
            oil_pressure = formatted_number_input(
                "Lubricating oil pressure in kPa",
                "0.001 to 10.0",
                minval=0.001,
                maxval=10.0,
                defvalue=3.300000,
                steps=0.001,
                valformat="%.6f"
            )
            
            
            fuel_pressure = formatted_number_input(
                "Fuel Pressure in kPa",
                "0.01 to 25.0",
                minval=0.01,
                maxval=25.0,
                defvalue=6.500000,
                steps=0.01,
                valformat="%.6f"
            )
    
        # define inputs in left column
        with col_right:
            coolant_pressure = formatted_number_input(
                "Coolant Pressure in kPa",
                "0.01 to 10.0",
                minval=0.01,
                maxval=10.0,
                defvalue=2.250000,
                steps=0.10,
                valformat="%.6f"
            )
            
            
            lub_oil_temp = formatted_number_input(
                "Lubricating oil Temperature in °C",
                "50.0 to 100.0",
                minval=50.0,
                maxval=100.0,
                defvalue=75.0,
                steps=0.1,
                valformat="%.6f"
            )
            
            
            coolant_temp = formatted_number_input(
                "Coolant Temperature in °C",
                "50.0 to 200.0",
                minval=50.0,
                maxval=200.0,
                defvalue=75.000000,
                steps=0.1,
                valformat="%.6f"
            )

        submitted = st.form_submit_button("🚀 Check Maintenance")


with col_output:

    # define place holders for output display
    output_placeholder = st.empty()
    probability_placeholder = st.empty()
    details_placeholder = st.empty()
    

    # ==========================
    # Single Value Prediction
    # ==========================
    with st.expander("🧠 Prediction Result", expanded=True):
        # dispaly result only after submit is done
        if submitted:
        
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
                "https://harishsohani-AIMLProjectTestBackEnd.hf.space/v1/EngPredMaintenance",
                json=input_data
                )
        
            if response.status_code == 200:
                ## get result as json
                result = response.json ()
        
                resp_status = result.get ("status")
        
                if resp_status == "success":
                    
                    ## Get Sales Prediction, probability Values
                    prediction_from_backend = result.get ("prediction")  # Extract only the value
                    probability = result.get ("probability")  # Extract only the value

                    # convert probability into % for representation
                    formatted_prob = f"{probability * 100:.2f}%"
            
                    # generate output string
                    if prediction_from_backend == 1:
                        output_placeholder.error("⚠️ Engine needs maintenance")
                    else:
                        output_placeholder.success("✅ Engine operating normally")

                    # dispaly probability of failur metric
                    probability_placeholder.metric("Failure Probability", formatted_prob)

                    # Display additional information
                    details_placeholder.markdown(f"""
                    *Model         :* XGBoost  
                    *Inference     :* Real-time
                    *Note*         : Probability of 50% and above is considered as Maintenance Needed.
                    """)
                    
                else:
        
                    error_str = result.get ("message")
    
                    output_placeholder.error(f"⚠️ {error_str}")
        
            elif response.status_code == 400 or response.status_code == 500:  # known errors
                
                ## get result as json
                result = response.json ()
    
                # get error message
                error_str = result.get ("message")
    
                # show error message
                output_placeholder.error(f"⚠️ Error processing request- Status Code : {response.status_code}, error : {error_str}")
                
            else:
                output_placeholder.error(f"⚠️ Error processing request- Status Code : {response.status_code}")
    

# ==============================
# Batch Prediction
# ==============================
st.markdown("---")

st.subheader ("Batch Prediction for Engine Maintenance")
st.markdown("""
*Select csv file with engine sensor parameters to find prediction for all readings*
""")

file = st.file_uploader ("Upload CSV file", type=["csv"])

if file is not None and st.button("🚀 Check Maintenance"):

    inputfile = {"file": (file.name, file.getvalue(), "text/csv")}
    response = requests.post(
        "https://harishsohani-AIMLProjectTestBackEnd.hf.space/v1/EngPredMaintenanceForBatch",
        files=inputfile
    )

    if response.status_code == 200:
        
        result = response.json ()

        resp_status = result.get ("status")
        
        if resp_status == "success":
                        
            ## Get Sales Prediction Value
            predictions_from_backend = result.get ("predictions")  # Extract only the value

            ## get probabbilities from back end
            probabilities = result.get ("probabilities")  # Extract only the value

            # read input data into data frame
            input_df = pd.read_csv(file)

            # Ensure lengths match
            if len(predictions_from_backend) == len(input_df):
    
                # Add prediction and probability column
                input_df["Prediction"]  = predictions_from_backend
                input_df["Probability"] = probabilities
    
                st.success("Batch prediction completed successfully")

                st.markdown("""
                *Prediction  : 1 deontes Maintenance is needed*
                """)

                st.markdown("""
                *Probability : This column indicates failure probability. Value ranges from 0 to 1. Value of 0.5 (50%) and above is considered as Maintenance Needed*
                """)

                st.markdown("""
                """)
    
                # Show combined dataframe
                st.dataframe(input_df, use_container_width=True)
    
            else:
                st.error("Prediction count does not match input records")            

            
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
