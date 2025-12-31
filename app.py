"""Streamlit chat app for Harbor Freight coupon assistant."""

import streamlit as st
from openai import OpenAI

from scraper import HarborFreightScraper


def load_coupons() -> str:
    """Scrape coupons and return LLM-formatted context."""
    scraper = HarborFreightScraper()
    coupons = scraper.scrape_all()
    return scraper.to_llm_context(coupons)


def get_system_prompt(coupon_context: str) -> str:
    """Build the system prompt with coupon context."""
    return f"""You are a helpful Harbor Freight coupon assistant. You have access to the current Harbor Freight coupons and can help users find deals.

When answering questions:
- Always include the coupon code, price, and expiration date when recommending items
- If multiple items match a query, list the top options
- Be concise but helpful
- If no coupons match what the user is looking for, say so honestly

{coupon_context}"""


def main():
    st.set_page_config(
        page_title="Harbor Freight Coupon Assistant",
        page_icon="🔧",
    )

    st.title("🔧 Harbor Freight Coupon Assistant")

    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except KeyError:
        st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
        st.stop()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Scrape coupons on startup (stored in session state)
    if "coupon_context" not in st.session_state:
        with st.spinner("Scraping Harbor Freight coupons..."):
            st.session_state.coupon_context = load_coupons()
        st.success("Coupons loaded! Ask me about deals.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about Harbor Freight deals..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build messages for API call
        api_messages = [
            {"role": "system", "content": get_system_prompt(st.session_state.coupon_context)},
            *st.session_state.messages,
        ]

        # Stream response
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_messages,
                stream=True,
            )
            response = st.write_stream(stream)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
