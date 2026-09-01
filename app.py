import json
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from chunker import split_into_chunks
from document_loader import LOADER_MAP, load_document
from run_rag_tests import run_tests
from vector_store import build_vector_store, reset_vector_store


SUPPORTED_DOCUMENT_TYPES = tuple(LOADER_MAP.keys())


def validate_test_cases(uploaded_file) -> list[dict]:
    try:
        test_cases = json.loads(uploaded_file.getvalue().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"The test-case file must be valid UTF-8 JSON: {error}") from error

    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError("The test-case JSON must be a non-empty list.")

    for index, test_case in enumerate(test_cases, start=1):
        if not isinstance(test_case, dict):
            raise ValueError(f"Test case {index} must be a JSON object.")
        missing_keys = {"user_input", "reference"} - test_case.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Test case {index} is missing required field(s): {missing}.")

    return test_cases


def save_upload(uploaded_file, directory: Path) -> Path:
    destination = directory / uploaded_file.name
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def build_index(specification_path: Path) -> int:
    reset_vector_store()
    documents = load_document(str(specification_path))
    chunks = split_into_chunks(documents)
    build_vector_store(chunks)
    return len(chunks)


st.set_page_config(page_title="RAG QA Runner", page_icon="R", layout="wide")
st.title("RAG QA Runner")
st.caption("Upload a specification and its matching test cases to build a fresh index and run grounded QA.")

with st.sidebar:
    st.subheader("Run settings")
    delay_seconds = st.number_input(
        "Delay between requests (seconds)", min_value=0.0, max_value=60.0, value=1.0, step=0.5
    )
    st.caption("A delay helps avoid free-tier API rate limits.")

left_column, right_column = st.columns(2)
with left_column:
    specification = st.file_uploader(
        "Specification document",
        type=[extension.removeprefix(".") for extension in SUPPORTED_DOCUMENT_TYPES],
        help="Supported: DOCX, PDF, TXT, Markdown, and CSV.",
    )
with right_column:
    test_case_file = st.file_uploader(
        "Test cases JSON", type=["json"], help="Each case needs user_input and reference fields."
    )

with st.expander("Expected test-case format"):
    st.code(
        json.dumps(
            [
                {
                    "user_input": "How do I reset my password?",
                    "reference": "Use the Forgot Password option on the sign-in screen.",
                }
            ],
            indent=2,
        ),
        language="json",
    )

if st.button("Build index and run tests", type="primary", disabled=not (specification and test_case_file)):
    try:
        test_cases = validate_test_cases(test_case_file)
        with tempfile.TemporaryDirectory(prefix="rag_qa_") as temporary_directory:
            specification_path = save_upload(specification, Path(temporary_directory))
            with st.status("Building a fresh vector index...", expanded=True) as status:
                chunk_count = build_index(specification_path)
                st.write(f"Indexed {chunk_count} chunks. Running {len(test_cases)} test cases...")
                evaluation_rows = run_tests(test_cases, sleep_between=delay_seconds)
                status.update(label="Test run complete", state="complete", expanded=False)

        st.session_state["evaluation_rows"] = evaluation_rows
        st.success(f"Completed {len(evaluation_rows)} test cases.")
    except Exception as error:
        st.error(f"The test run could not complete: {error}")

if evaluation_rows := st.session_state.get("evaluation_rows"):
    st.subheader("Results")
    st.dataframe(
        [
            {
                "Question": row["user_input"],
                "Response": row["response"],
                "Reference": row["reference"],
                "Retrieved chunks": len(row["retrieved_contexts"]),
            }
            for row in evaluation_rows
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download evaluation dataset",
        data=json.dumps(evaluation_rows, indent=2),
        file_name="eval_dataset.json",
        mime="application/json",
    )