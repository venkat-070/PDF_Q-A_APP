import streamlit as st
import tempfile
from rag_pipeline_V2 import ask_question,process_pdf
import re

st.title("PDF DOCUMENT Q&A")
st.caption("Upload a pdf and ask question about it.")

uploaded_file = st.file_uploader("Upload a pdf",type="pdf")

if uploaded_file is None:
    st.info("upload a pdf to get started..")
else:
    if "pdf_processed" not in st.session_state:
        
        with st.spinner("Processing pdf...."):
            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            collection_name = uploaded_file.name.rsplit(".", 1)[0]
            collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)
            collection_name = collection_name[:60]
                
            st.session_state["Collection_name"] = collection_name
            process_pdf(tmp_path,st.session_state["Collection_name"])
            st.session_state["pdf_processed"] = True
    st.success(f"📄 Loaded: {st.session_state['Collection_name']}")
    Question = st.text_input("Ask your question: ")
    
    if st.button("Ask"):
        if Question:
            with st.spinner("Thinking....."):
                answer , pages = ask_question(Question,st.session_state["Collection_name"])
                st.write(answer.content)
            st.info(f"📖 Source pages: {', '.join(str(p) for p in pages)}")
        else:
            st.warning("Please enter a question")


