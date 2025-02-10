import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os
from dotenv import load_dotenv
from difflib import get_close_matches  # Fuzzy matching

load_dotenv()

# Initialize the LLM with Groq API
groq_api = os.getenv("groq_api")
llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api)

# Streamlit UI
st.title("Excel Query Assistant with LLM")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    # Load all sheets into a dictionary
    xls = pd.ExcelFile(uploaded_file)
    sheet_dfs = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}
    sheet_names = list(sheet_dfs.keys())

    
   

    # Query input
    user_query = st.text_area("Enter your query about the data:")

    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Identifying relevant sheet..."):
                # Ask the LLM to determine the best sheet
                sheet_selection_prompt = (
                    "You are analyzing an Excel file with multiple sheets.\n"
                    f"The available sheets are: {sheet_names}\n"
                    "Based on the following query, determine which sheet contains the most relevant data:\n"
                    f"Query: {user_query}\n"
                    "Return ONLY the exact sheet name from the list above. No extra words."
                )

                response = llm.invoke(sheet_selection_prompt)
                selected_sheet = response.content.strip()  # Extract response text

                st.write(f"**LLM Response:** `{selected_sheet}`")  # Debugging output

                # Use fuzzy matching to find the closest matching sheet name
                matched_sheet = get_close_matches(selected_sheet, sheet_names, n=1, cutoff=0.6)
                if matched_sheet:
                    selected_sheet = matched_sheet[0]
                else:
                    st.error("LLM couldn't determine a valid sheet. Please refine your query.")
                    st.stop()

            st.success(f"Using sheet: {selected_sheet}")
            df = sheet_dfs[selected_sheet]
            st.write("### Preview of the selected sheet:")
            st.dataframe(df.head())

            # Create a Pandas agent for the selected sheet
            agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

            with st.spinner("Processing query..."):
                response = agent.run(user_query)
                st.write("### Answer:")
                st.write(response)
        else:
            st.warning("Please enter a query.")
