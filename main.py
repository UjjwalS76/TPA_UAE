import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.memory import ConversationBufferMemory
import json

# Configure page
st.set_page_config(
    page_title="UAE Transfer Pricing Assessment Tool",
    layout="wide",
    page_icon="🔍"
)

# Initialize LLM with Perplexity API
@st.cache_resource
def initialize_llm():
    """Initialize the LLM with Perplexity API"""
    try:
        # Get API key from secrets
        if "PPLX_API_KEY" not in st.secrets:
            st.error("Perplexity API key not found. Please check your secrets configuration.")
            st.stop()
        
        api_key = st.secrets["PPLX_API_KEY"]
        model = st.secrets.get("MODEL", "llama-3.1-sonar-small-128k-online")
        
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base="https://api.perplexity.ai/v1",  # Updated API endpoint
            temperature=0,
            streaming=True
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.stop()

# Define output schemas
assessment_schema = ResponseSchema(
    name="assessment",
    description="Detailed assessment of the relationship between the parties"
)

relationship_schema = ResponseSchema(
    name="relationship_type",
    description="The specific type of relationship identified"
)

basis_schema = ResponseSchema(
    name="basis",
    description="The legal and factual basis for the determination"
)

risk_schema = ResponseSchema(
    name="risk_level",
    description="Risk level assessment (HIGH, MEDIUM, or LOW)"
)

documentation_schema = ResponseSchema(
    name="documentation",
    description="Required documentation and compliance requirements"
)

output_parser = StructuredOutputParser.from_response_schemas([
    assessment_schema,
    relationship_schema,
    basis_schema,
    risk_schema,
    documentation_schema
])

# System prompt
SYSTEM_PROMPT = """You are an expert AI assistant specializing in UAE Transfer Pricing regulations and Related Party determinations. 
Your task is to analyze relationships between parties and determine if they qualify as Related Parties or Connected Persons under UAE TP rules.

Key considerations:
1. Family relationships up to 4th degree
2. Ownership/control thresholds of 50% or more
3. Direct and indirect relationships
4. Special cases like listed companies and regulated entities
5. Risk assessment and documentation requirements

Please provide your analysis in a structured format covering:
1. Clear assessment of the relationship
2. Specific relationship type identified
3. Legal and factual basis for determination
4. Risk level assessment
5. Documentation requirements

{format_instructions}
"""

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_input}")
])

def process_relationship_assessment(party1_details, party2_details):
    """Process the relationship assessment using LangChain and LLM"""
    try:
        llm = initialize_llm()
        
        # Format input
        assessment_input = f"""
        Party 1 Details:
        {json.dumps(party1_details, indent=2)}
        
        Party 2 Details:
        {json.dumps(party2_details, indent=2)}
        
        Please analyze if these parties are Related Parties or Connected Persons under UAE Transfer Pricing rules.
        """
        
        # Get formatted response
        formatted_prompt = prompt.format_messages(
            user_input=assessment_input,
            format_instructions=output_parser.get_format_instructions()
        )
        
        response = llm.predict_messages(formatted_prompt)
        return output_parser.parse(response.content)
        
    except Exception as e:
        st.error(f"Error in processing assessment: {str(e)}")
        return None

def main():
    st.title("🔍 UAE Transfer Pricing Related Party Assessment Tool")
    
    # Debug information (only shown if DEBUG is enabled in secrets)
    if st.secrets.get("DEBUG", "false").lower() == "true":
        with st.expander("Debug Information"):
            st.write("Configuration:")
            st.write(f"- API Base: https://api.perplexity.ai/v1")
            st.write(f"- Model: {st.secrets.get('MODEL', 'Not Set')}")
            st.write("- API Key Status: " + ("Configured" if "PPLX_API_KEY" in st.secrets else "Missing"))
            
            if st.button("Test API Connection"):
                try:
                    llm = initialize_llm()
                    response = llm.predict("test")
                    st.success("API Connection Successful!")
                except Exception as e:
                    st.error(f"API Connection Failed: {str(e)}")
    
    # Add sidebar with info
    with st.sidebar:
        st.header("About")
        st.info("""
        This tool helps determine Related Party status under UAE Transfer Pricing rules.
        
        Enter details for both parties and get:
        - Relationship assessment
        - Risk level
        - Documentation requirements
        """)
        
        st.header("References")
        st.write("""
        - UAE Federal Decree-Law No. 47 of 2022
        - UAE Transfer Pricing Guidelines
        - OECD Transfer Pricing Guidelines
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["Assessment Tool", "Help & Instructions"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Party 1 Details")
            party1_type = st.selectbox(
                "Type of Party 1",
                ["Individual", "Company"],
                key="party1_type"
            )
            
            party1_details = {}
            if party1_type == "Individual":
                party1_details.update({
                    "type": "Individual",
                    "name": st.text_input("Name", key="party1_name"),
                    "residency": st.selectbox(
                        "Residency Status",
                        ["UAE Resident", "Non-Resident"],
                        key="party1_residency"
                    )
                })
            else:
                party1_details.update({
                    "type": "Company",
                    "name": st.text_input("Company Name", key="party1_name"),
                    "company_type": st.selectbox(
                        "Company Type",
                        ["LLC", "Branch", "Free Zone Entity", "Other"],
                        key="party1_company_type"
                    ),
                    "listed_status": st.checkbox("Listed on Recognized Stock Exchange", key="party1_listed"),
                    "regulated": st.checkbox("Under UAE Regulatory Oversight", key="party1_regulated")
                })
        
        with col2:
            st.subheader("Party 2 Details")
            party2_type = st.selectbox(
                "Type of Party 2",
                ["Individual", "Company"],
                key="party2_type"
            )
            
            party2_details = {}
            if party2_type == "Individual":
                party2_details.update({
                    "type": "Individual",
                    "name": st.text_input("Name", key="party2_name"),
                    "residency": st.selectbox(
                        "Residency Status",
                        ["UAE Resident", "Non-Resident"],
                        key="party2_residency"
                    )
                })
            else:
                party2_details.update({
                    "type": "Company",
                    "name": st.text_input("Company Name", key="party2_name"),
                    "company_type": st.selectbox(
                        "Company Type",
                        ["LLC", "Branch", "Free Zone Entity", "Other"],
                        key="party2_company_type"
                    ),
                    "listed_status": st.checkbox("Listed on Recognized Stock Exchange", key="party2_listed"),
                    "regulated": st.checkbox("Under UAE Regulatory Oversight", key="party2_regulated")
                })
        
        # Relationship details
        st.subheader("Relationship Details")
        if party1_type == "Company" and party2_type == "Company":
            col1, col2, col3 = st.columns(3)
            with col1:
                ownership_pct = st.slider("Ownership Percentage", 0, 100, 0)
                party1_details["ownership_percentage"] = ownership_pct
            with col2:
                voting_rights = st.slider("Voting Rights Percentage", 0, 100, 0)
                party1_details["voting_rights"] = voting_rights
            with col3:
                board_control = st.checkbox("Board Control")
                party1_details["board_control"] = board_control
        
        elif party1_type == "Individual" and party2_type == "Individual":
            relationship = st.selectbox(
                "Family Relationship",
                [
                    "None",
                    "Parent/Child",
                    "Grandparent/Grandchild",
                    "Sibling",
                    "Uncle/Aunt/Nephew/Niece",
                    "First Cousin",
                    "Other"
                ]
            )
            party1_details["family_relationship"] = relationship
        
        # Process button
        if st.button("Analyze Relationship", type="primary"):
            if not (party1_details.get("name") and party2_details.get("name")):
                st.error("Please fill in all required fields")
                return
            
            with st.spinner("Analyzing relationship..."):
                result = process_relationship_assessment(party1_details, party2_details)
                
                if result:
                    # Display results in an expander
                    with st.expander("Assessment Results", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Assessment")
                            st.write(result.assessment)
                            st.write(f"**Relationship Type:** {result.relationship_type}")
                            st.write(f"**Basis:** {result.basis}")
                        
                        with col2:
                            st.markdown("### Risk and Documentation")
                            st.write(f"**Risk Level:** {result.risk_level}")
                            st.markdown("**Required Documentation:**")
                            st.write(result.documentation)
    
    with tab2:
        st.markdown("""
        ### How to Use This Tool
        
        1. Select the type for each party (Individual or Company)
        2. Fill in the required details
        3. For companies, specify ownership and control details
        4. For individuals, specify family relationships if applicable
        5. Click 'Analyze Relationship' to get the assessment
        
        ### Understanding Results
        
        The tool will provide:
        - Relationship status determination
        - Specific type of relationship identified
        - Legal basis for the determination
        - Risk level assessment
        - Required documentation
        """)

if __name__ == "__main__":
    main()
