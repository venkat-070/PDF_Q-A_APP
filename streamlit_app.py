import os
import re
import tempfile
import streamlit as st

from rag_pipeline_V2 import ask_question, process_pdf

st.title("PDF DOCUMENT Q&A")
st.caption("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is None:
    st.info("Upload a PDF to get started.")
else:

    if (
        "current_pdf" not in st.session_state
        or st.session_state["current_pdf"] != uploaded_file.name
    ):

        with st.spinner("Processing PDF..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            collection_name = uploaded_file.name.rsplit(".", 1)[0]
            collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)[:60]

            st.session_state["Collection_name"] = collection_name

            process_pdf(tmp_path, collection_name)

            os.remove(tmp_path)

            st.session_state["current_pdf"] = uploaded_file.name
            st.session_state["pdf_processed"] = True

    st.success(f"📄 Loaded: {st.session_state['Collection_name']}")

    question = st.text_input("Ask your question:")

    if st.button("Ask"):
        if question:
            with st.spinner("Thinking..."):
                answer, pages = ask_question(
                    question,
                    st.session_state["Collection_name"]
                )

                # If ask_question() returns output.content
                st.write(answer)

                # If ask_question() returns output instead, use:
                # st.write(answer.content)

            pages = sorted(set(pages))
            st.info(f"📖 Source pages: {', '.join(map(str, pages))}")

        else:
            st.warning("Please enter a question.")
