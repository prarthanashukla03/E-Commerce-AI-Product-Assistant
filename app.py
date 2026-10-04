import streamlit as st

from rag import generate_answer


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="E-Commerce RAG Assistant",
    page_icon="🛒",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🛒 E-Commerce AI Product Assistant")

st.write(
    "Ask questions about products using "
    "Retrieval-Augmented Generation (RAG)."
)


# ==========================================
# USER QUESTION
# ==========================================

query = st.text_input(
    "Ask your question:",
    placeholder="Example: Suggest a laptop under ₹70000 for programming"
)


# ==========================================
# ASK AI BUTTON
# ==========================================

if st.button("Ask AI"):

    if query.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Searching products and generating answer..."
        ):

            answer, products = generate_answer(
                query
            )


        # ======================================
        # AI ANSWER
        # ======================================

        st.subheader("🤖 AI Answer")

        st.write(answer)


        # ======================================
        # RETRIEVED PRODUCTS
        # ======================================

        st.subheader("📚 Retrieved Product Information")

        for i, product in enumerate(products):

            with st.expander(
                f"Product {i + 1}"
            ):

                st.write(product)