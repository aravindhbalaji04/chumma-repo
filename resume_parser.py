import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import json
from ats_score import calculate_score  # Ensure this function accepts parsed JSON text
import re

# 🧠 Configure Google Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyDfzvb_QnYsh0S_0Yk4LWVcvTiU936zjuo"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 📝 Gemini Prompt Template
PROMPT_TEMPLATE = """
You are an intelligent resume parser designed to extract detailed candidate information in a precise structured JSON format.

From the raw resume text below, extract and organize the following data:

Return the output in *exactly* this JSON format (with accurate nesting and field names):

{
  "Skills": {
    "Languages": [...],
    "Technologies": [...],
    "Core": [...]
  },
  "Certifications": [
    ...
  ],
  "Projects": [
    {
      "title": "...",
      "date": "...",
      "details": [
        "...",
        "..."
      ]
    },
    ...
  ],
  "Work Experience": [
    {
      "role": "...",
      "organization": "...",
      "location": "...", // if not available, use null
      "date": "...",
      "responsibilities": [
        "...",
        "..."
      ]
    },
    ...
  ]
}

Rules:
- Classify "Skills" into three categories: Languages, Technologies, Core (conceptual/academic).
- Each "Project" should include a title, date (if available), and bullet point descriptions.
- "Work Experience" must include role, organization, date range, location (null if not mentioned), and responsibilities as bullet points.
- Only return the JSON object. No explanations, headings, or comments.
- If any field has no data, use an empty list [] or null appropriately.

Now, extract from the resume below:

{resume_text}
"""

# 📤 Extract text from PDF using PyMuPDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    resume_text = "\n".join(page.get_text() for page in doc)
    return resume_text

# 🤖 Ask Gemini to parse the resume
def parse_resume_with_gemini(resume_text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = PROMPT_TEMPLATE.replace("{resume_text}", resume_text)
    response = model.generate_content(prompt)
    # Clean any code block backticks (```json or ```)
    clean_text = re.sub(r"^```(?:json)?|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
    return clean_text


# 🚀 Streamlit App
st.title("📄 ATS SCORE")
st.markdown("Upload a resume PDF")

uploaded_file = st.file_uploader("Drag and drop a resume PDF here", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("🔍 Extracting text from PDF..."):
        resume_text = extract_text_from_pdf(uploaded_file)

    with st.spinner("🤖 Parsing resume with Gemini..."):
        parsed_json_text = parse_resume_with_gemini(resume_text)

    try:
        parsed_output = json.loads(parsed_json_text)
        st.success("✅ Resume successfully parsed!")
        st.subheader("🧾 Parsed Resume (JSON):")
        st.json(parsed_output)

        st.subheader("📊 ATS Score:")
        ats_score = calculate_score(parsed_output)
        st.write(f"**Score:** {ats_score}/100")

    except Exception as e:
        st.error("⚠️ Failed to parse JSON output.")
        st.text(f"Raw Output:\n{parsed_json_text}")
        st.text(f"Error: {e}")
