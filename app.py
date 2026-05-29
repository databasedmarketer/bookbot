"""
Streamlit web interface for the BookBot RAG Assistant
"""

import re
import streamlit as st
import time
from typing import Dict, Any, List
from rag_pipeline import BookRAGPipeline
from llm_client import send_transcript_via_ses
from config import PAGE_TITLE, PAGE_ICON

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_icon=PAGE_ICON,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #600043;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .metric-box { display: none; }
    .action-buttons {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .stChatInput > div { border-color: #9e9e9e !important; }
    .stChatInput textarea:focus {
        border-color: #9e9e9e !important;
        box-shadow: 0 0 0 1px #9e9e9e !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Exclusion-detection helpers ────────────────────────────────────────────────

# Phrases that signal the user wants to exclude books
_EXCLUSION_TRIGGERS = [
    r"don'?t recommend",
    r"do not recommend",
    r"not interested in",
    r"don'?t show",
    r"do not show",
    r"skip\b",
    r"exclude\b",
    r"don'?t want to see",
    r"do not want to see",
    r"remove\b.*\bbook",
    r"no more\b.*\bbook",
    r"i don'?t want (these|those|this) book",
    r"not (these|those|this) book",
    r"hide (these|those|this)",
]
_EXCLUSION_PATTERN = re.compile("|".join(_EXCLUSION_TRIGGERS), re.IGNORECASE)


def _detect_exclusion_intent(text: str) -> bool:
    """Return True if the user's message looks like an explicit book exclusion."""
    return bool(_EXCLUSION_PATTERN.search(text))


def _extract_book_titles_from_history(messages: list) -> List[str]:
    """
    Pull book titles that BookBot has already mentioned in the session by
    looking for common Markdown heading patterns in assistant messages.
    E.g. '## The Lean Startup' or '**Title:** The Lean Startup'
    """
    titles: List[str] = []
    heading_re = re.compile(
        r"(?:#{1,3}\s+(.+?)$|"           # ## Book Title
        r"\*\*Title:\*\*\s*(.+?)(?:\*\*|$)|"   # **Title:** Book Title
        r"\*\*Book:\*\*\s*(.+?)(?:\*\*|$))",   # **Book:** Book Title
        re.MULTILINE,
    )
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for m in heading_re.finditer(msg.get("content", "")):
            title = next(g for g in m.groups() if g is not None).strip()
            if title and title not in titles:
                titles.append(title)
    return titles


def _format_transcript(messages: list) -> str:
    """Convert the session message list to a plain-text transcript."""
    lines = []
    for msg in messages:
        role_label = "You" if msg["role"] == "user" else "BookBot"
        lines.append(f"{role_label}:\n{msg['content']}\n")
    return "\n".join(lines)


# ── Pipeline caching ───────────────────────────────────────────────────────────

@st.cache_resource
def initialize_pipeline(cache_version: int = 5):
    """Initialize the RAG pipeline (cached for performance)"""
    try:
        pipeline = BookRAGPipeline()
        if pipeline.initialize():
            return pipeline
        return None
    except Exception as e:
        import traceback
        print(f"❌ Exception during pipeline initialization: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None


# ── Source display helper ──────────────────────────────────────────────────────

def display_chat_message(role: str, content: str, sources: List[Dict] = None):
    """Display a chat message with proper formatting"""
    is_fr = ("french_mode" in st.session_state and st.session_state.french_mode)
    if role == "user":
        st.markdown(f"""
        <div class="query-box">
            <strong>👤 {'Vous' if is_fr else 'You'}:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="answer-box">
            <strong>🤖 Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

        if sources:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    metadata = source.get('metadata', {})
                    relevance = source.get('relevance_score', 0)
                    if is_fr:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {i}</strong> (Pertinence : {relevance:.3f})<br>
                            <strong>Titre:</strong> {metadata.get('book_title', 'N/A')}<br>
                            <strong>Catégorie:</strong> {metadata.get('book_category', 'N/A')}<br>
                            <strong>Description:</strong> {source.get('document', '')[:200]}...
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {i}</strong> (Relevance: {relevance:.3f})<br>
                            <strong>Book Title:</strong> {metadata.get('book_title', 'N/A')}<br>
                            <strong>Category:</strong> {metadata.get('book_category', 'N/A')}<br>
                            <strong>Description:</strong> {source.get('document', '')[:200]}...
                        </div>
                        """, unsafe_allow_html=True)


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    is_fr_header = ("french_mode" in st.session_state and st.session_state.french_mode)

    st.markdown(f"""
    <div class="main-header">
        {PAGE_ICON}
    </div>
    """, unsafe_allow_html=True)

    # Initialize pipeline
    with st.spinner(
        "🚀 Initialisation du système RAG..."
        if ("french_mode" in st.session_state and st.session_state.french_mode)
        else "🚀 Initializing RAG system..."
    ):
        pipeline = initialize_pipeline(cache_version=4)

    if pipeline is None:
        is_fr = ("french_mode" in st.session_state and st.session_state.french_mode)
        st.error(
            "❌ Échec de l'initialisation du système RAG. Vérifiez votre configuration."
            if is_fr
            else "❌ Failed to initialize the RAG system. Please check your setup."
        )
        with st.expander("🔧 Troubleshooting / Dépannage", expanded=True):
            st.markdown("""
            **Common issues / Problèmes courants:**

            1. **Gemini API key not set / Clé API Gemini manquante**
               - Export key: `export GOOGLE_API_KEY=...`

            2. **Missing data files / Fichiers de données manquants**
               - Ensure CSV files exist in `data/` directory

            3. **Check terminal output / Vérifiez la sortie du terminal**
            """)
        st.stop()

    # ── Session-state initialisation ───────────────────────────────────────────
    if "french_mode" not in st.session_state:
        st.session_state.french_mode = False
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = True
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.5
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Books the user has explicitly asked to exclude this session
    if "excluded_books" not in st.session_state:
        st.session_state.excluded_books: List[str] = []
    # Persist the transcript recipient email across reruns
    if "transcript_email" not in st.session_state:
        st.session_state.transcript_email = ""
    # Track last send result so it survives reruns
    if "transcript_send_result" not in st.session_state:
        st.session_state.transcript_send_result = None  # (success, msg) or None

    is_fr = st.session_state.french_mode
    selected_language = "fr" if is_fr else "en"

    # ── Top control bar ────────────────────────────────────────────────────────
    st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
    cols = st.columns([2, 1, 2, 2, 2])

    with cols[0]:
        if st.button("Nettoyage" if is_fr else "Cleanup", key="clear_chat_btn"):
            st.session_state.messages = []
            st.session_state.excluded_books = []
            st.rerun()

    with cols[1]:
        pass  # French toggle placeholder

    with cols[2]:
        pass  # RAG toggle placeholder

    with cols[3]:
        st.session_state.temperature = st.slider(
            "Température" if is_fr else "Temp",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.temperature),
            step=0.05,
            help=(
                "Température du modèle : plus bas = plus déterministe, plus haut = plus créatif."
                if is_fr
                else "LLM temperature: lower is more deterministic, higher is more creative."
            ),
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Email transcript panel ─────────────────────────────────────────────────
    with st.expander(
        "📧 Envoyer la transcription par e-mail" if is_fr else "📧 Email this transcript",
        expanded=False,
    ):
        if not st.session_state.messages:
            st.info(
                "Aucun message à envoyer pour l'instant."
                if is_fr
                else "No messages to send yet. Start a conversation first."
            )
        else:
            # Use session state so the address survives st.rerun() calls
            def _on_email_change():
                st.session_state.transcript_email = st.session_state._email_widget
                st.session_state.transcript_send_result = None  # clear stale result

            st.text_input(
                "Votre adresse e-mail" if is_fr else "Your email address",
                value=st.session_state.transcript_email,
                placeholder="you@example.com",
                key="_email_widget",
                on_change=_on_email_change,
            )

            email_ready = bool(st.session_state.transcript_email.strip())
            send_btn = st.button(
                "Envoyer" if is_fr else "Send transcript",
                key="send_transcript_btn",
                disabled=not email_ready,
            )

            if send_btn and email_ready:
                # Clear any previous result before sending
                st.session_state.transcript_send_result = None
                transcript_text = _format_transcript(st.session_state.messages)
                with st.spinner("Envoi en cours..." if is_fr else "Sending…"):
                    result = send_transcript_via_ses(
                        transcript=transcript_text,
                        recipient_email=st.session_state.transcript_email.strip(),
                        subject=(
                            "Votre transcription BookBot"
                            if is_fr
                            else "Your BookBot Chat Transcript"
                        ),
                    )
                st.session_state.transcript_send_result = result

            # Show last send result (persists across reruns until next send/edit)
            if st.session_state.transcript_send_result is not None:
                success, msg = st.session_state.transcript_send_result
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    # ── Chat history ───────────────────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Chat input ─────────────────────────────────────────────────────────────
    placeholder_text = (
        "Posez des questions sur les livres et leurs résumés :"
        if selected_language == "fr"
        else "Ask about books and their summaries:"
    )
    user_input = st.chat_input(placeholder_text)

    if user_input:
        # ── Detect explicit exclusion intent ───────────────────────────────────
        if _detect_exclusion_intent(user_input):
            # If the user says something like "I don't want to see these books",
            # pull every book title BookBot has mentioned so far and exclude them.
            # If they name a specific book ("don't recommend Atomic Habits"),
            # we also capture that below via the LLM's own awareness of the
            # instruction; but we add the recently-surfaced titles as a safety net.
            recently_mentioned = _extract_book_titles_from_history(
                st.session_state.messages
            )
            for title in recently_mentioned:
                if title not in st.session_state.excluded_books:
                    st.session_state.excluded_books.append(title)

            # Also do a lightweight scan of the user's message for a quoted or
            # capitalised book title (e.g. "skip 'Atomic Habits' please")
            quoted_re = re.compile(r'[""\'"]([A-Z][^"""\'"\n]{2,60})[""\'"]')
            for m in quoted_re.finditer(user_input):
                candidate = m.group(1).strip()
                if candidate not in st.session_state.excluded_books:
                    st.session_state.excluded_books.append(candidate)

        # ── Save and display the user message ─────────────────────────────────
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── Generate assistant response ────────────────────────────────────────
        if st.session_state.rag_enabled:
            with st.spinner(
                "🔍 Bookbot réfléchit..." if selected_language == "fr"
                else "🔍 Bookbot is thinking..."
            ):
                try:
                    result = pipeline.query(
                        user_input,
                        language=selected_language,
                        temperature=float(st.session_state.temperature),
                        excluded_books=list(st.session_state.excluded_books),
                    )
                    assistant_text = result.get(
                        'answer', 'Error: No answer returned from query.'
                    )
                except Exception as e:
                    assistant_text = f"Error processing query: {str(e)}"
        else:
            with st.spinner(
                "💬 Génération de la réponse..." if selected_language == "fr"
                else "💬 Generating response..."
            ):
                try:
                    assistant_text = pipeline.llm_client.chat_general(
                        user_input,
                        language=selected_language,
                        temperature=float(st.session_state.temperature),
                    )
                except TypeError:
                    assistant_text = pipeline.llm_client.chat_general(
                        user_input,
                        language=selected_language,
                    )

        # ── Save and display the assistant message ─────────────────────────────
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_text}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_text)

        st.rerun()


if __name__ == "__main__":
    main()