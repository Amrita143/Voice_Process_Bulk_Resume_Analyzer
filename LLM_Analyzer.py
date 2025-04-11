import os
from dotenv import load_dotenv
import json
import openai
from groq import Groq
load_dotenv()
import re
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

client = Groq(api_key= st.secrets["LLMA_API_KEY"])

# LLM Resume Analysis by two LLMs one after another, first one will analyze and second one will return structured output
def llm_resume_analysis(resume_extracted_text):
    prompt = f""" 
    Below is the resume details of the applicant. \n
    {resume_extracted_text}
"""
    # First LLM will analyze the resume and return the analysis
    completion = client.chat.completions.create(
    model="deepseek-r1-distill-llama-70b",
    messages=[
        {
            "role": "system",
            "content": "## Role and Task: \nYou are a **Senior Hiring Manager** in a **BPO company specializing in US debt collection** (based in India). Your task is to **review, analyze, and evaluate** applicants' resumes based on the specified hiring requirements. You must categorize candidates into one of three levels and provide a clear explanation of your decision.\n\n## Hiring Requirement:\nRole: Voice Agent  \nProcess: International voice processes (e.g., debt collection or loan services)\n\n## Candidate Categorization:\n\n1) **Good (Highly Preferred):**  \nExperience: Prior work experience in international voice processes (preferably in debt collection).  \nAdded Advantage: Experience in recognized BPO or debt collection firms, such as: Astra Business Services Private Limited, GLOBAL VANTEDGE, iQor, IDC Technologies, Provana, Encore Capital Group, American Express (Amex), Genpact, iEnergizer, Teleperformance, Personiv, Fusion CX, HCLTech, Barclays MIS, Accenture  \n\n2) **Average (Moderately Suitable):**  \nMinimum 6 months of work experience in sales or customer service roles (any domain).\n\n3) **Non-Qualified (Unsuitable):**  \n- Less than 6 months of relevant experience or  \n- Fresher (no prior work experience) or  \n- Resume does not align with voice process job requirements.\n\n**Special Considerations/Remarks:** Identify applicants from Northeast India, specifically from any of these states: Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Tripura\n\n## Evaluation Criteria & Explanation:\n- Categorize the candidate into one of the three levels based on their resume.  \n- Justify the classification with a brief explanation, mentioning relevant experience, skills, and suitability for the role.  \n- Add special remarks if applicable (Northeast India origin).  \n\n**Additional Information:**  \n- Mention the candidate's **name**, **mobile number** (if available in the resume, otherwise N/A), and **email** (if available in the resume, otherwise N/A). \n\n---\n\n**Important:**  \nDo not invent or assume any details, Don't make things up by yourself. All information, analysis, and evaluation provided must strictly be based on the resume data given as input. If no resume data is provided, fill all the required fields (such as name, mobile number, email, category, justification etc.) as **\"N/A\"**."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.6,
    top_p=0.95,
    stream=False,
    stop=None,
    )

    analyzer_llm_response = completion.choices[0].message.content

    # Remove the <think> tag and its content using a regex
    cleaned_response = re.sub(r'<think>.*?</think>', '', analyzer_llm_response, flags=re.DOTALL)

    # Optionally, strip extra whitespace from the result
    cleaned_response = cleaned_response.strip()

    # pass the cleaned_response to the second llm for structured output generation
    response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "# Role, Goal and Task : \nYou are an LLM agent tasked with extracting key information from candidate resume descriptions. Given the candidate details, your job is to parse the text and return a JSON object that strictly follows the provided schema. The resume description may include evaluation details such as candidate name, mobile number, email, categorization, justification, and any special remarks.\n\n #### Important:Do not invent or assume any details, Don't make things up by yourself. All information and extracted details must strictly be based on the resume data given as input. If no resume data is provided, fill all the required fields/json keys (such as name, mobile number, email, category, justification etc.) as **N/A**.\n\n ## Requirements:\n1. Extract the candidate's name and assign it to the key \"name\".\n2. Extract the candidate's mobile number and assign it to the key \"mobile\".  If mobile number not found in resume, mention \"N/A\".\n3. Extract the candidate's email and assign it to the key \"email\".  If email not found in resume, mention \"N/A\".\n4. Extract the categorization information and map it to the key \"category\". The value must be one of the following:\n   - \"unsuitable\" (for non-qualified candidates)\n   - \"average\" (for moderately suitable candidates)\n   - \"good\" (for highly preferred candidates)\n5. Extract the justification details and assign them to the key \"justification\".\n6. Extract any special remarks and assign them to the key \"special_remarks\". The value must be either \"northeast\" or \"other_state\".  \"special_remarks\" describes where the candidate is from. Incase, if its not mentioned clearly or not found in resume, always choose \"other_state\" value. \n7. Return only a valid JSON object with these keys and no additional information.\n8. Do not include any commentary, explanations, or extra text in the output.\n\n ## Example expected JSON output:\n{\n  \"name\": \"Rahul Sharma\",\n  \"mobile\": \"8910463080\",\n  \"email\": \"N/A\",\n  \"category\": \"unsuitable\",\n  \"justification\": \"Rahul Sharma's resume indicates 2 years and 4 months of experience, primarily as a Business Analyst at Astra Business Services Private Limited. However, his role focused on data analysis and process optimization rather than direct voice process or debt collection experience. His sales experience, though relevant, was only 4 months, which is insufficient to meet the Average category's requirement of at least 6 months in sales or customer service.\",\n  \"special_remarks\": \"other_state\"\n}\n\nEnsure that the output JSON exactly follows this structure and contains no extra keys.\n"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": cleaned_response
        }
      ]
    }
  ],
  response_format={
    "type": "json_schema",
    "json_schema": {
      "name": "candidate_resume",
      "strict": True,
      "schema": {
        "type": "object",
        "required": [
          "name",
          "mobile",
          "email",
          "category",
          "justification",
          "special_remarks"
        ],
        "properties": {
          "name": {
            "type": "string",
            "description": "The name of the candidate."
          },
          "mobile": {
            "type": "string",
            "description": "The mobile number of the candidate."
          },
          "email": {
            "type": "string",
            "description": "The email of the candidate."
          },
          "category": {
            "enum": [
              "unsuitable",
              "average",
              "good"
            ],
            "type": "string",
            "description": "Categorization of the candidate's suitability."
          },
          "justification": {
            "type": "string",
            "description": "Details explaining the categorization of the candidate."
          },
          "special_remarks": {
            "enum": [
              "northeast",
              "other_state"
            ],
            "type": "string",
            "description": "Notes regarding the candidate's location."
          }
        },
        "additionalProperties": False
      }
    }
  },

    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    structured_output = response.choices[0].message.content
    # json string to dictionary
    structured_output = json.loads(structured_output)
    return structured_output
# returns dictionary
