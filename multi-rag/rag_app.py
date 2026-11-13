
import os
import tempfile
import hashlib

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Assistant")
st.caption("Ask questions across multiple PDF documents with page-level sources.")


# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------

if "store" not in st.session_state:
    st.session_state.store = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# --------------------------------------------------
# 3. API KEY VALIDATION
# --------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY is not configured. "
        "Please configure it in your terminal."
    )
    st.stop()


# --------------------------------------------------
# 4. PDF UPLOAD
# --------------------------------------------------

st.header("📂 Upload your PDF documents")

uploaded_files = st.file_uploader(
    "Select multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# 5. INDEX ALL PDF DOCUMENTS
# --------------------------------------------------

if st.button("🚀 Build Knowledge Base", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:

        all_chunks = []
        indexed_names = set()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        progress = st.progress(0)
        status = st.empty()

        try:
            for index, uploaded_file in enumerate(uploaded_files):

                status.info(
                    f"Processing {uploaded_file.name}..."
                )

                # Save PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:
                    loader = PyPDFLoader(temp_path)
                    pages = loader.load()

                    # Preserve the original PDF filename
                    for page in pages:
                        page.metadata["source"] = uploaded_file.name

                    # Split document into chunks
                    chunks = splitter.split_documents(pages)

                    all_chunks.extend(chunks)
                    indexed_names.add(uploaded_file.name)

                finally:
                    os.remove(temp_path)

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            if not all_chunks:
                st.error("No readable text was found in the PDFs.")
                st.stop()

            status.info("Creating embeddings and FAISS index...")

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )

            vector_store = FAISS.from_documents(
                all_chunks,
                embeddings
            )

            # Store everything in Streamlit session state
            st.session_state.store = vector_store
            st.session_state.indexed_files = sorted(indexed_names)
            st.session_state.chunk_count = len(all_chunks)

            status.success("Knowledge base created successfully!")

            st.success(
                f"Indexed {len(all_chunks)} chunks "
                f"from {len(indexed_names)} PDF documents."
            )

        except Exception as e:
            st.error(f"Indexing failed: {str(e)}")


# --------------------------------------------------
# 6. DISPLAY INDEXED DOCUMENTS
# --------------------------------------------------

if st.session_state.store is not None:

    st.divider()
    st.header("📊 Knowledge Base Status")

    col1, col2 = st.columns(2)

    col1.metric(
        "Total PDF Documents",
        len(st.session_state.indexed_files)
    )

    col2.metric(
        "Total Indexed Chunks",
        st.session_state.chunk_count
    )

    st.write("**Indexed documents:**")

    for filename in st.session_state.indexed_files:
        st.write(f"📄 {filename}")


# --------------------------------------------------
# 7. QUESTION ANSWERING
# --------------------------------------------------

if st.session_state.store is not None:

    st.divider()
    st.header("💬 Ask your documents")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: What is the leave policy?"
    )

    if st.button("🔍 Get Answer"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching all uploaded PDFs..."):

                try:
                    retriever = st.session_state.store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 6}
                    )

                    docs = retriever.invoke(question)

                    if not docs:
                        st.warning("No relevant information found.")

                    else:

                        # Prepare source references
                        context_parts = []

                        for doc in docs:
                            filename = doc.metadata.get(
                                "source", "Unknown file"
                            )

                            page_number = doc.metadata.get(
                                "page", "Unknown"
                            )

                            # PyPDFLoader page numbers start at 0
                            if isinstance(page_number, int):
                                page_number += 1

                            context_parts.append(
                                f"""
                                Source: {filename}
                                Page: {page_number}
                                Content:
                                {doc.page_content}
                                """
                            )

                        context = "\n\n".join(context_parts)

                        # Ask model to identify conflicting evidence
                        prompt = ChatPromptTemplate.from_messages([
                            (
                                "system",
                                """
                                You are a document question-answering assistant.

                                Answer only using the supplied PDF context.

                                Rules:
                                1. Do not invent information.
                                2. If the documents disagree, explicitly
                                   explain the disagreement.
                                3. Identify which document supports each
                                   conflicting statement.
                                4. If evidence is insufficient, say so.
                                5. Keep the answer clear and concise.
                                6. Do not claim that all documents agree
                                   unless the retrieved evidence supports it.

                                PDF CONTEXT:
                                {context}
                                """
                            ),
                            ("human", "{question}")
                        ])

                        llm = ChatOpenAI(
                            model="gpt-4o-mini",
                            temperature=0,
                            api_key=api_key
                        )

                        chain = prompt | llm

                        response = chain.invoke({
                            "context": context,
                            "question": question
                        })

                        st.subheader("📝 Answer")

                        st.write(response.content)

                        # Display source documents
                        st.subheader("📚 Sources")

                        unique_sources = {}

                        for doc in docs:

                            filename = doc.metadata.get(
                                "source", "Unknown file"
                            )

                            page_number = doc.metadata.get(
                                "page", "Unknown"
                            )

                            if isinstance(page_number, int):
                                page_number += 1

                            source_key = (filename, page_number)

                            unique_sources[source_key] = doc.page_content

                        for filename, page in unique_sources:

                            st.info(
                                f"📄 **{filename}** — Page {page}"
                            )

                except Exception as e:
                    st.error(f"Question answering failed: {str(e)}")


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Python | Streamlit | LangChain | OpenAI | FAISS"
)
