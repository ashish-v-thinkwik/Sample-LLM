from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd

@st.cache_data
def load_excel_sheets(uploaded_file):
    """Loads all sheets from an Excel file into a dictionary of DataFrames."""
    xls = pd.ExcelFile(uploaded_file)
    sheet_data = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
    return sheet_data

def main():
    load_dotenv()

    # Load OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not set in the environment variables.")
        return

    st.set_page_config(page_title="Ask Your Excel Data")
    st.header("📊 Ask Your Excel Data with AI")

    # File uploader
    uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1]

        if file_type == "xlsx":
            sheet_data = load_excel_sheets(uploaded_file)
            selected_sheet = st.selectbox("Select a sheet to analyze", list(sheet_data.keys()))
            df = sheet_data[selected_sheet]
        else:
            df = pd.read_csv(uploaded_file)
            selected_sheet = "CSV File"

        

        # ✅ Use LangChain's Pandas Agent for querying
        llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
        agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)

        user_question = st.text_input("Ask a question about your data:")

        if user_question:
            # ✅ Improve AI Prompting by specifying the sheet name
            improved_question = f"Using data from the sheet '{selected_sheet}', answer: {user_question}"

            with st.spinner("🔍 Finding the answer..."):
                response = agent.run(improved_question)
                st.write(f"**📄 Answer from: {selected_sheet}**")
                st.write(response)

if __name__ == "__main__":
    main()
