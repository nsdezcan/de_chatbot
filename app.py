import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# -------------------------------------------------
# 1. SAYFA KONFİGÜRASYONU VE STİL
# -------------------------------------------------
st.set_page_config(page_title="Karriere-Chatbot", page_icon="🤖", layout="wide")

# Google Drive yolları kaldırıldı, yerine göreli (relative) yollar eklendi.
# Bu dosyaların "assets" adında bir klasörde olduğunu varsayıyoruz.
LOGO_BA_PATH = "assets/logo_ba.png"
LOGO_COMPANY_PATH = "assets/logo_company.png"

def set_styles():
    st.markdown("""
        <style>
        .stApp { background-color: #f0f2f6; }
        [data-testid="chat-messages"] { background-color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #ffffff; }
        .st-emotion-cache-16txtl3 { padding-top: 2rem; }
        </style>
    """, unsafe_allow_html=True)

set_styles()

# -------------------------------------------------
# 2. DİL VE METİN YÖNETİMİ
# -------------------------------------------------
LANGUAGES = {
    "Deutsch": {
        "page_title": "Karriere-Chatbot",
        "sidebar_title": "Einstellungen",
        "main_title": "Agentur für Arbeit",
        "main_subtitle": "Karriere-Assistent",
        "chat_placeholder": "Stellen Sie hier Ihre Frage...",
        "language_select": "Sprache",
        "welcome_message": "Hallo! Wie kann ich Ihnen bei Ihrer Karriereplanung helfen?",
        "error_key": "Gemini API-Schlüssel nicht gefunden. Bitte fügen Sie ihn in der Seitenleiste hinzu.",
        "info_source": "Quelle",
        "no_vector": "📦 Vectorstore nicht gefunden. Ich antworte nur mit LLM.",
    },
    "English": {
        "page_title": "Career Chatbot",
        "sidebar_title": "Settings",
        "main_title": "Federal Employment Agency",
        "main_subtitle": "Career Assistant",
        "chat_placeholder": "Ask your question here...",
        "language_select": "Language",
        "welcome_message": "Hello! How can I assist you with your career planning today?",
        "error_key": "Gemini API key not found. Please add it in the sidebar.",
        "info_source": "Source",
        "no_vector": "📦 Vectorstore not found. I will answer only with LLM.",
    },
}

# -------------------------------------------------
# 3. CHATBOT ÇEKİRDEK FONKSİYONLARI
# -------------------------------------------------

@st.cache_resource
def get_embeddings_model():
    """Embedding modelini bir kez yükler ve cache'ler."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore(_embeddings):
    """Vektör veritabanını bir kez yükler ve cache'ler."""
    
    # Google Drive yolu kaldırıldı, "vectorstore" klasörünün ana dizinde olduğunu varsayıyoruz.
    vectorstore_path = "vectorstore" 
    
    if os.path.exists(vectorstore_path):
        return FAISS.load_local(
            vectorstore_path,
            _embeddings,
            allow_dangerous_deserialization=True, # Güvenlik uyarısı için bu parametre gerekli
        )
    return None

def get_context_from_vectorstore(vectorstore, question, k=4):
    """Soruyu vektör veritabanında arar ve ilgili context'i döndürür."""
    if vectorstore is None:
        return "", []
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    context_text = "\n\n".join([d.page_content for d in docs])
    return context_text, docs

def ask_gemini(gemini_api_key: str, question: str, context: str) -> str:
    """Gemini modeline context ile birlikte soruyu sorar."""
    
    # API anahtarı yoksa ortam değişkeninden (Secrets) almayı dener
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        return "Hata: Gemini API anahtarı bulunamadı."

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
    You are a helpful assistant. Use the context below to answer the question.
    If the answer is not in the context, say you don't know.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        st.error(f"Gemini API ile iletişimde bir hata oluştu: {str(e)}")
        return "Bir hata nedeniyle cevap veremiyorum."

# -------------------------------------------------
# 4. STREAMLIT ARAYÜZÜ
# -------------------------------------------------

# API Anahtarını Streamlit Secrets'tan (ortam değişkeni) okumayı dener
api_key_from_env = os.getenv("GEMINI_API_KEY")

if "language" not in st.session_state:
    st.session_state.language = "Deutsch"

texts = LANGUAGES[st.session_state.language]

with st.sidebar:
    if os.path.exists(LOGO_COMPANY_PATH):
        st.image(LOGO_COMPANY_PATH, width=150)
    st.title(texts["sidebar_title"])

    selected_language = st.selectbox(
        texts["language_select"],
        options=list(LANGUAGES.keys()),
        index=0 if st.session_state.language == "Deutsch" else 1,
    )
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()

    # Kullanıcının manuel API anahtarı girmesine izin verir
    # Varsayılan değer olarak Secrets'tan okunan anahtarı kullanır
    gemini_api_key = st.text_input(
        "Gemini API Key", type="password", value=api_key_from_env or ""
    )

col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists(LOGO_BA_PATH):
        st.image(LOGO_BA_PATH, width=150)
with col2:
    st.title(texts["main_title"])
    st.subheader(texts["main_subtitle"])

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": texts["welcome_message"]}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(texts["chat_placeholder"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not gemini_api_key:
        with st.chat_message("assistant"):
            st.error(texts["error_key"])
    else:
        # Modelleri ve veritabanını yükle (cache sayesinde hızlı çalışır)
        try:
            embeddings = get_embeddings_model()
            vectorstore = load_vectorstore(embeddings)
        except Exception as e:
            vectorstore = None
            st.error(f"Vektör veritabanı yüklenirken hata oluştu: {e}")

        # Context'i al
        if vectorstore:
            with st.spinner("Dokümanlar aranıyor..."):
                context_text, _ = get_context_from_vectorstore(vectorstore, prompt)
        else:
            context_text = ""
            st.info(texts["no_vector"])

        # Gemini'den cevabı al
        answer = ask_gemini(gemini_api_key, prompt, context_text)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
