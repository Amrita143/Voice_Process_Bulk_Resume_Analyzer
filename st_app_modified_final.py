import os
import io
import zipfile
import time
import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from llama_cloud_services import LlamaParse
from LLM_Analyzer import llm_resume_analysis

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase = create_client(supabase_url, supabase_key)

# Initialize LlamaParse
LLAMA_CLOUD_API_KEY = st.secrets["LLAMA_CLOUD_API_KEY"]
parser = LlamaParse(
    result_type="markdown" # Options: "markdown" or "text
)

# Set page config
st.set_page_config(
    page_title="Bulk Resume Processor",
    page_icon="📄",
    layout="wide"
)

def process_resume(file_link, parser=None):
    """Process a single 'resume file public http link' and return extracted text"""
    if parser is None:
        parser = LlamaParse(
            result_type="markdown",
            api_key=LLAMA_CLOUD_API_KEY
        )
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # st.write(f"Processing attempt {attempt+1} for {file_link}")
            documents = parser.load_data(file_link)
            
            # Check if documents is empty
            if not documents:
                st.warning(f"Warning: No documents returned for {file_link}")
                if attempt < max_retries - 1:
                    st.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return ""
            
            # Collect the text from the documents
            extracted_text = ""
            for doc in documents:
                if hasattr(doc, 'text') and doc.text:
                    extracted_text += doc.text + "\n\n"  # add a new line after each document
            
            # If text was successfully extracted, return it
            if extracted_text and len(extracted_text.strip()) > 10:
                return extracted_text
            else:
                st.warning(f"Warning: Empty or very short text extracted from {file_link}")
                if attempt < max_retries - 1:
                    st.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                return ""
                
        except Exception as e:
            st.error(f"Error in attempt {attempt+1}: {str(e)}")
            if attempt < max_retries - 1:
                st.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                st.error(f"Failed after {max_retries} attempts for {file_link}")
                return ""
    
    return ""

def upload_to_supabase_storage(file_data, folder_name, file_name):
    """Upload a file to Supabase storage and return the public URL"""
    
    # Set the correct content type based on file extension
    file_extension = file_name.split('.')[-1]

    content_type = None
    if file_extension.lower() == '.pdf':
        content_type = 'application/pdf'
    elif file_extension.lower() == '.doc':
        content_type = 'application/msword'
    elif file_extension.lower() == '.docx':
        content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    else:
        content_type = 'application/octet-stream'

    # Full path in storage including folder
    storage_path = f"{folder_name}/{file_name}"
    
    # Upload to Supabase storage
    response = supabase.storage.from_("bulk_resumes").upload(
        storage_path,
        file_data,
        {"content-type": content_type} 
    )
    
    # Get the public URL
    public_url = supabase.storage.from_("bulk_resumes").get_public_url(storage_path)
    
    return public_url

def save_to_supabase_db(resume_data, resume_url):
    """Save resume data to Supabase database"""
    data = {
        "name": resume_data["name"],
        "mobile": resume_data["mobile"],
        "email": resume_data["email"],
        "resume_url": resume_url,
        "candidate_category": resume_data["category"],
        "special_remarks": resume_data["special_remarks"],
        "justification": resume_data["justification"]
    }
    
    # Insert data into the bulk_applicants table
    response = supabase.table("bulk_applicants").insert(data).execute()
    
    # Return the ID of the inserted record
    return response.data[0]["id"] if response.data else None

def clear_supabase_table():
    """Clear all records from the bulk_applicants table"""
    try:
        # Delete all records from the table
        response = supabase.table("bulk_applicants").delete().neq('id', 0).execute()
        return True
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")
        return False

def get_all_applicants_data():
    """Retrieve all records from the bulk_applicants table"""
    response = supabase.table("bulk_applicants").select("*").execute()
    return response.data

def main():
    st.title("Voice Process- AI Resume Processor")
    # add a streamlit expander about AI-instruction
    with st.expander("AI-Instruction"):
        st.write("""
                    ## Role and Task:

                    You are a **Senior Hiring Manager** in a **BPO company specializing in US debt collection** (based in India). Your task is to **review, analyze, and evaluate** applicants' resumes based on the specified hiring requirements. You must categorize candidates into one of three levels and provide a clear explanation of your decision.

                    ## Hiring Requirement:

                    - **Role:** Voice Agent  
                    - **Process:** International voice processes (e.g., debt collection or loan services or retail or credit card)

                    ## Candidate Categorization:

                    ### 1) **Good (Highly Preferred):**
                    - **Experience:** Prior work experience in international voice processes (preferably in debt collection).
                    - **Added Advantage:** Experience in recognized BPO or debt collection firms, such as Astra Business Services Private Limited, GLOBAL VANTEDGE, iQor, IDC Technologies, Provana, Encore Capital Group, American Express (Amex), Genpact, iEnergizer,Teleperformance, Personiv, Fusion CX, HCLTech, Barclays MIS, Accenture etc. 

                    ### 2) **Average (Moderately Suitable):**
                    - Minimum 6 months of work experience in sales or customer service roles (any domain).

                    ### 3) **Non-Qualified (Unsuitable):**
                    - Less than 6 months of relevant experience, or
                    - Fresher (no prior work experience), or
                    - Resume does not align with voice process job requirements.

                    ## Special Considerations/Remarks:
                    - Identify applicants from Northeast India, specifically from these states: Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Tripura.

                    ## Evaluation Criteria & Explanation:
                    - Categorize the candidate into one of the three levels based on their resume.
                    - Justify the classification with a brief explanation, mentioning relevant experience, skills, and suitability for the role.
                    - Include special remarks if applicable (Northeast India origin).

                    ## Additional Information:
                    - Mention the candidate's **name**, **mobile number** (if available in the resume, otherwise N/A), and **email** (if available in the resume, otherwise N/A).

                    ---

                    **Important:**  
                    Do not invent or assume any details, Don't make things up by yourself. All information, analysis, and evaluation provided must strictly be based on the resume data given as input. If no resume data is provided, fill all the required fields (such as name, mobile number, email, category, justification etc.) as **"N/A"**.

                 """)
        

    st.write("Upload a ZIP file containing resumes to process and analyze them.")
    
    uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
    
    if uploaded_file:
        # Create a unique folder name for Supabase storage
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = os.path.splitext(uploaded_file.name)[0]
        storage_folder_name = f"{zip_filename}_{timestamp}"


        # Read the uploaded file as bytes and wrap into a BytesIO
        try:
            zip_bytes = uploaded_file.read()
            zip_io = io.BytesIO(zip_bytes)

            # Open the zip file
            with zipfile.ZipFile(zip_io, 'r') as zip_ref:
                # Filter .pdf, .doc, .docx files
                resume_files = [file for file in zip_ref.infolist() if file.filename.lower().endswith(('.pdf', '.doc', '.docx'))]

                if resume_files:
                    st.write(f"Found {len(resume_files)} resume files. Uploading...")

                    resume_public_urls = []
                    for file_info in resume_files:
                        file_data = zip_ref.read(file_info.filename)
                        # upload to supabase storage
                        resume_url = upload_to_supabase_storage(file_data, storage_folder_name, file_info.filename)
                        resume_public_urls.append({"file_name": file_info.filename, "url": resume_url})

                    st.write(f"Successfully uploaded {len(resume_public_urls)} resumes to storage.")

                    # Clear the existing data in the table
                    if clear_supabase_table():
                        st.info("Cleared existing data from the database.")
                    
                    # Now process each resume from its public URL
                    process_progress = st.progress(0)
                    process_status = st.empty()
                    process_status.info("Processing resumes...")
                    
                    # Process each resume file
                    results = []
                    success_count = 0
                    error_count = 0
                    
                    for i, resume_data in enumerate(resume_public_urls):
                        file_name = resume_data["file_name"]
                        resume_url = resume_data["url"]
                        process_status.info(f"[{i+1}/{len(resume_public_urls)}] Processing {file_name}...")
                        
                        try:
                            # Re-initialize parser for each file to avoid session issues
                            file_parser = LlamaParse(
                                result_type="markdown",  # Options: "markdown" or "text"
                                api_key=LLAMA_CLOUD_API_KEY  # Ensure API key is passed explicitly
                            )
                            
                            # Extract text from the resume using the public URL
                            
                            extracted_text = process_resume(resume_url, file_parser)  # Pass parser as parameter
                    

                            if not extracted_text or len(extracted_text.strip()) < 10:
                                raise ValueError(f"Insufficient text extracted from file")
                            
                            # Analyze the resume using LLM
                            result = llm_resume_analysis(extracted_text)
                            
                            # Validate the result contains required fields
                            required_fields = ["name", "mobile", "email", "category", "special_remarks", "justification"]
                            missing_fields = [field for field in required_fields if field not in result or not result[field]]
                            
                            if missing_fields:
                                raise ValueError(f"LLM analysis missing required fields: {', '.join(missing_fields)}")
                            
                            # Save the data to Supabase database
                            record_id = save_to_supabase_db(result, resume_url)
                            
                            # Add to results
                            results.append(result)
                            success_count += 1
                            process_status.success(f"✅ Successfully processed {file_name}")
                            
                        except Exception as e:
                            error_count += 1
                            error_msg = f"❌ Error processing {file_name}: {str(e)}"
                            process_status.error(error_msg)
                        
                        # Update progress bar
                        process_progress.progress((i + 1) / len(resume_public_urls))
                        
                        # Add a small delay between files to avoid rate limiting
                        if i > 0:  # Don't delay before the first file
                            time.sleep(1)  # 1 second pause between files
                    
                    # Final status
                    process_status.empty()
                    st.success(f"Processing complete! Successfully processed {success_count} resumes with {error_count} errors.")
                    
                    # Get all data from the database
                    applicants_data = get_all_applicants_data()
                    
                    if applicants_data:
                        # Convert to DataFrame
                        df = pd.DataFrame(applicants_data)
                        
                        # Display the data
                        st.subheader("Processed Resumes")
                        st.dataframe(df)
                        
                        # Create a download button for CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"resume_analysis_{timestamp}.csv",
                            mime="text/csv"
                        )
                        
                        # Check for potential data issues
                        null_counts = df.isnull().sum()
                        if null_counts.sum() > 0:
                            st.warning("⚠️ Warning: The following columns have null values:")
                            st.write(null_counts[null_counts > 0])
                    else:
                        st.warning("No data was saved to the database.")
                else:
                    st.warning("No resume files found in the ZIP file.")
        except Exception as e:
            st.error(f"Error processing ZIP file: {str(e)}")

if __name__ == "__main__":
    main()