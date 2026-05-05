import os
import time
import tempfile
import uuid

import streamlit as st
import pandas as pd 
import plotly.express as px

from rag_pipeline import build_pipeline, compute_relevance_score
from logger import log_query, get_session_stats


st.set_page_config(
    page_title="FinSight | Financial Document Assistant | by Madhushree",
    page_icon="📑",
    layout="wide",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline_stats" not in st.session_state:
    st.session_state.pipeline_stats = None

# ----Sidebar-----------------------------------------------------------------------------------

with st.sidebar:
    st.title("📑 FinSight")
    st.caption("Ask anything from your financial documents — no more manual searching")
    st.divider()

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload Financial PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} document(s) loaded")
        for f in uploaded_files:
            size_kb = round(len(f.getvalue()) / 1024, 1)
            st.caption(f"· {f.name} ({size_kb} KB)")

    st.divider()
    st.markdown("**Try asking:**")
    examples = [
        "What is the net profit?",
        "What are the key risk factors?",
        "Summarise the CEO message.",
        "What is dividend per share?",
        "What does MAS require here?"
    ]
    for q in examples:
        if st.button(q, use_container_width = True, key=q):
            st.session_state.pending_question = q 

    st.divider()
    st.caption("Built by **Madhushree Varatharaj**")
    st.caption("[LinkedIn](https://www.linkedin.com/in/madhushree-varatharaj-9b7308204/) · [GitHub](https://github.com/MadhushreeVaratharaj)")
    st.caption(f"Session: '{st.session_state.session_id}' ")

# ---Tabs---------------------------------------------------------------------------
 
tab_chat, tab_monitor, tab_about = st.tabs(["Chat", "Monitoring", "About"])

# ----- Tab 1: Chat ----------------------------------------------------------------

with tab_chat:

    if not uploaded_files:
        st.info("Hi! Upload a financial PDF from the sidebar and start asking questions — no manual searching needed.")
        st.markdown("""
        **What you can do:**
        - Query annual reports for revenue, profit, dividends
        - Ask what MAS requires in regulatory circulars
        - Extract key terms from loan agreements
        - Cross-reference multiple documents at once
        """)

    else:
        @st.cache_resource(show_spinner="Building document index...")
        def get_pipeline (cache_key):
            temp_paths=[]
            for f in uploaded_files:
                f.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    temp_paths.append(tmp.name)
            return build_pipeline(temp_paths)

        cache_key = tuple((f.name, len(f.getvalue())) for f in uploaded_files)
        qa_chain, pipeline_stats = get_pipeline(cache_key)
        st.session_state.pipeline_stats = pipeline_stats

        with st.expander("Pipeline stats"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Documents", pipeline_stats["num_documents"])
            c2.metric("Pages", pipeline_stats["num_pages"])
            c3.metric("Chunks", pipeline_stats["num_chunks"])
            c4.metric("Index Time", f"{pipeline_stats['index_time_s']}s")

        st.divider()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for src in msg["sources"]:
                            st.markdown(f"**{src['file']}** - Page {src['page']}")
                            st.caption(src["excerpt"])

        question = st.chat_input("What do you want to know from your documents?")
        if not question and "pending_question" in st.session_state:
            question = st.session_state.pop("pending_question")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching documents.."):
                    t0 = time.time()
                    result = qa_chain({"query": question})
                    latency = time.time() - t0
                answer = result["result"]
                sources = result["source_documents"]
                relevance_score = compute_relevance_score(question, sources)

                st.markdown(answer)

                source_meta =[]
                source_pages = []
                for doc in sources:
                    fname = os.path.basename(doc.metadata.get("Source", "Unkown"))
                    page = doc.metadata.get("page", 0) + 1
                    excerpt = doc.page_content[:250].strip() + "..."
                    source_meta.append({"file": fname, "page": page, "excerpt": excerpt})
                    source_pages.append(page)

                if source_meta:
                    with st.expander(f"Sources ({len(source_meta)} chunks used)"):
                        for src in source_meta:
                            st.markdown(f"**{src['file']}** — Page {src['page']}")
                            st.caption(src["excerpt"])
                            st.divider()

                colour = "green" if latency <3 else "orange" if latency < 6 else "red"
                rel_colour = "green" if relevance_score > 0.7 else "orange" if relevance_score > 0.5 else "red"
                st.markdown(
                    f'<span style="font-size:11px;color:{colour}"> 🕐 {latency:2f}s</span>'
                    f' &nbsp;&nbsp; <span style="font-size:11px;color:{rel_colour}">relevance {relevance_score:.2f}</span>',
                    unsafe_allow_html = True,
                )
                st.caption(
                    "Relevance score reflects how many retrieved chunks matched your query keywords. "
                    "1.0 = all 4 chunks matched · 0.5 = 2 out of 4 · 0.0 = no match found."
                )

            log_query(
                question = question,
                answer = answer,
                latency_s = latency,
                num_sources = len(source_meta),
                source_pages = source_pages,
                session_id = st.session_state.session_id,
                relevance_score = relevance_score,
            )

            st.session_state.messages.append({
                "role" : "assistant",
                "content": answer,
                "sources": source_meta,
            })

#------- Tab 2 : Monitoring ------------------------------------------------------------------------

with tab_monitor:
    st.header("Monitoring Dashboard")
    st.caption("Every query is tracked — latency, sources used, and timestamps")

    log = st.session_state.get("query_log", [])

    if not log:
        st.info("Ask questions in the Chat tab to see metrics here.")
    else:
        stats = get_session_stats()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total queries", stats["total_queries"])
        c2.metric("Avg latency", f"{stats['avg_latency_s']}s")
        c3.metric("Peak latency", f"{stats['max_latency_s']}s")
        c4.metric("Avg sources used", stats["avg_sources"])
        c5.metric("Avg relevance", stats["avg_relevance"] if stats.get("avg_relevance") else "-")

        st.divider()

        df = pd.DataFrame(log)
        df["query_num"] = range(1, len(df) + 1)

        fig1 = px.line(
            df, x="query_num", y="latency_s",
            title = "Response latency per query",
            labels = {"query_num": "Query #", "latency_s": "Latency (s)"},
            markers = True,
        )
        fig1.update_layout(height=260, margin=dict(t=40, b=20))
        st.plotly_chart(fig1, use_container_width = True)

        fig2 = px.bar(
            df, x="query_num", y="num_sources",
            title="Chunks retrieved per query",
            labels = {"query_num": "Query #", "num_sources": "Chunks"},
            color="num_sources",
            color_continuous_scale="Blues",
        )
        fig2.update_layout(height=240, margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width = True)

        st.subheader("Query log")
        display = df[["timestamp", "question", "latency_s", "num_sources"]].copy()
        display["question"] = display["question"].str[:60] + "..."
        st.dataframe(display,use_container_width=True)

    if st.session_state.pipeline_stats:
        st.divider()
        st.subheader("Pipeline config")
        st.json(st.session_state.pipeline_stats)


#----- Tab 3: About -------------------------------------------------------------------------------

with tab_about:
    st.header("About FinSight")
    st.markdown("""
    
    FinSight is a RAG-based application for querying financial documents using natural language.
    During my time at Cognizant as a Programmer Analyst, I worked on Temenos T24 — 
    a core banking system used by financial institutions globally. Learning the system meant 
    going through large technical documents and the Temenos Learning Community (TLC), 
    Temenos's official knowledge base. TLC had hundreds of pages covering modules, 
    transaction workflows, field definitions, and system configurations. Finding one specific 
    answer meant manually searching through all of it — slow, frustrating, and inefficient, 
    especially when you're new and under pressure to deliver.
    
    When I started my MSc in Data Science at SUTD and learned about Retrieval-Augmented Generation, 
    that experience came back to me immediately. FinSight is what I wish I had at Cognizant — 
    upload the document, ask your question in plain English, get the answer with the exact page 
    it came from. No more manual searching.
    
    I built it in a banking context deliberately. MAS circulars, compliance notices, annual reports, 
    and technical banking documentation are exactly the kind of dense, high-stakes documents 
    where getting the right information quickly actually matters.
    
    **How it Works**

    Upload PDF 

    -> PyPDF loader 

    -> text chunking (1000 characters/150 overlap)

    -> MiniLM-L6-v2 embeddings 

    -> FAISS index

    -> top-4 retrieval 

    -> Llama 3.1 8B Instant (via Groq) 

    -> Cited answer with page reference
    
    **Why these choices?**
    
    Chunk size 1000 with 150 overlap — financial and technical documents have long, 
    dense clauses. Smaller chunks lose context mid-sentence, larger chunks dilute retrieval relevance.
    
    MiniLM-L6-v2 — lightweight, runs fully local, no document data leaves the machine during indexing. 
    Important in banking where document confidentiality is non-negotiable.
    
    temperature=0 — financial Q&A needs deterministic output. Any randomness risks 
    returning inconsistent figures or dates for the same question.
    
    FAISS — no infrastructure overhead. Fast enough for document-scale retrieval 
    without needing a dedicated vector database.

    RAG - Financial documents change constantly = new MAS circulars, quarterly reports, updated loan terms.
    RAG means no retraining needed. Swap the document, Knowledge updates instantly.

    **Stack**
    - LLM: Llama 3.1 8B Instant (via Groq)
    - Embeddings: sentence-transformers/all-MiniLM-L6-v2
    - Vector store: FAISS
    - Framework: LangChain
    - Frontend: Streamlit
    """)
