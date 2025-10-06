import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# Konfigurasi Awal dan Styling

st.set_page_config(
    page_title="AutoBOT: Agen Otomotif ReAct",
    page_icon="🚗",
    layout="wide"
)

st.title("🤖 AutoBOT: Asisten Otomotif ReAct")
st.caption("Informasi Otomotif real-time dan ReAct untuk penalaran.")

# CSS Kustom untuk tema Otomotif
st.markdown("""
    <style>
    /* Mengganti warna background chat untuk tema 'Midnight Black' atau gelap */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    /* Styling tombol untuk tema otomotif */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        border: 2px solid #FF4B4B;
    }
    .stButton>button:hover {
        background-color: #C00000;
        border-color: #C00000;
    }
    /* Chat message styling */
    .st-emotion-cache-r421h3, .st-emotion-cache-gq0p5k {
        background-color: #333333;
        border-radius: 10px;
    }
    .st-emotion-cache-gq0p5k {
        background-color: #004B8D;
    }
    /* Mengatur gaya link teks di sidebar agar terlihat seperti teks biasa */
    .link-text {
        font-size: 14px;
        color: #FF4B4B;
        cursor: pointer;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar untuk Pengaturan dan API Key

try:
    TAVILY_API_KEY_LOADED = st.secrets["TAVILY_API_KEY"]
except KeyError:
    TAVILY_API_KEY_LOADED = None

try:
    GEMINI_API = st.secrets["GEMINI_API"]
except KeyError:
    GEMINI_API = None

with st.sidebar:
    st.markdown(
        """
        <h1 style="
            text-align: center;
            font-weight: 700;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        ">
            🚗 AutoBOT
        </h1>
        """, unsafe_allow_html=True
    )

    google_api_key = GEMINI_API

    st.markdown("---")

    # Tombol reset
    reset_button = st.button("🔄 Reset Percakapan", help="Hapus semua pesan dan mulai sesi baru")

    # Tombol tentang
    if st.button("💡 Tentang", help="Tentang pengembang AutoBOT"):
        st.session_state["new_prompt_to_process"] = "Tentang pengembang AutoBOT"
        st.rerun()

    st.markdown("---")

    # ===== Statistik & Status =====
    if "messages" not in st.session_state:
        st.session_state.messages = []

    total_messages = len(st.session_state.messages) // 2 # Hitung jumlah pasangan user/assistant
    st.subheader("📊 Status Sistem")
    if google_api_key and TAVILY_API_KEY_LOADED:
        st.success(f"🟢 Sistem Stabil — Semua API siap!")
    else:
        st.error("🔴 Sistem Tidak Aktif — Periksa API Key")

    st.metric(label="💬 Total Pertanyaan Hari Ini", value=total_messages)

    system_load = min(total_messages * 10, 100)

    st.markdown("---")

    st.markdown(
        """
        💬 **Kata kata Hari Ini:**
        > "AI terbaik bukan yang menggantikan manusia,
        > tapi yang membantu manusia menjadi lebih hebat."
        """
    )

    st.markdown("---")
    st.markdown("""
            * [Instagram](https://www.instagram.com/a_bidillah)
            * [Github](https://github.com/ewbidd)
            * [LinkedIn](https://linkedin.com/in/muhammad-abidillah)
              """)

    st.markdown("---")
    st.caption("© 2025 AutoBOT — Dibangun dengan ❤️ oleh Muhammad Abidillah")

# Definisi Tools & Agent Initialization

SYSTEM_PROMPT = """
Anda adalah AutoBOT, seorang spesialis dan asisten otomotif yang ramah dan antusias.
Tugas Anda adalah memberikan informasi yang akurat, terperinci, dan terkini tentang mobil, motor, perawatan, spesifikasi, modifikasi, dan masalah teknis kendaraan.

Anda **HARUS** menggunakan alat 'tavily_search_results_json' untuk mendapatkan informasi FAKTUAL, real-time, atau berita terbaru.
Gunakan penalaran (Thought) untuk merencanakan pencarian yang efektif.

Jika pengguna bertanya selain dari otomotif, Anda **HARUS** menolak pertanyaan tersebut dengan sopan menggunakan respons standar berikut:
**RESPONS STANDAR PENOLAKAN:** "Maaf, saya adalah chatbot yang dikhususkan untuk menjawab pertanyaan seputar otomotif saja. Silakan ajukan pertanyaan seputar mobil, motor, perawatan, spesifikasi, modifikasi, atau masalah teknis kendaraan."

Jawablah dengan nada yang profesional namun penuh semangat. Berikan jawaban dalam format yang mudah dibaca.
"""

def initialize_agent(g_key, t_key):
    try:
        os.environ["TAVILY_API_KEY"] = t_key
        tavily_search_tool = TavilySearchResults(max_results=3)
        tools = [tavily_search_tool]

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=g_key,
            temperature=0.4
        )

        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SYSTEM_PROMPT
        )
        return agent
    except Exception as e:
        os.environ.pop("TAVILY_API_KEY", None)
        st.error(f"Gagal menginisialisasi Agen. Error: {e}")
        return None

# Inisialisasi Logic & Error Handling

if not google_api_key or not TAVILY_API_KEY_LOADED:
    if not google_api_key:
        st.warning("⚠️ Mohon masukkan Google AI API Key Anda di sidebar untuk memulai.", icon="🗝️")
    if not TAVILY_API_KEY_LOADED:
        st.stop()
    st.stop()

current_key = google_api_key
if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != current_key):
    with st.spinner("⏳ Menginisialisasi AutoBOT dan Tavily Search Tool..."):
        agent_instance = initialize_agent(google_api_key, TAVILY_API_KEY_LOADED)

    if agent_instance:
        st.session_state.agent = agent_instance
        st.session_state._last_key = current_key
        # st.session_state.pop("messages", None)
        st.success("AutoBOT Siap! Pencarian web aktif 🚀")
    else:
        st.error("Gagal menginisialisasi Agen. Silakan periksa kunci API Gemini Anda.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if reset_button:
    st.session_state.pop("agent", None)
    st.session_state.pop("messages", None)
    st.session_state.pop("_last_key", None)
    os.environ.pop("TAVILY_API_KEY", None)
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if "new_prompt_to_process" in st.session_state and st.session_state["new_prompt_to_process"]:

    prompt_to_process = st.session_state.pop("new_prompt_to_process")

    # Lakukan pemrosesan pesan khusus
    if prompt_to_process == "Tentang pengembang AutoBOT":

        # Tambahkan prompt user ke chat
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Jawaban yang di hardcode
        special_answer = """
Halo, saya adalah **AutoBOT**, Asisten AI otomotif berbasis LLM menggunakan model **Gemini 2.5 Flash**.

Saya dikembangkan oleh **Muhammad Abidillah**, seorang mahasiswa Teknik Informatika Universitas Riau.

Jika ingin berkenalan lebih lanjut, kamu bisa menemukan saya di:
* [Instagram](https://www.instagram.com/a_bidillah)
* [Github](https://github.com/ewbidd)
* [LinkedIn](https://linkedin.com/in/muhammad-abidillah)

Kamu bisa menemukan repository GitHub dari projek ini [disini](https://github.com/ewbidd/autobot).
"""

        # Tampilkan jawaban assistant (dari special_answer)
        with st.chat_message("assistant"):
            st.markdown(special_answer)
        st.session_state.messages.append({"role": "assistant", "content": special_answer})

        st.rerun()

prompt_from_input = st.chat_input("Tanyakan tentang mobil, modifikasi, atau tips perawatan...")

if prompt_from_input:

    st.session_state.messages.append({"role": "user", "content": prompt_from_input})
    with st.chat_message("user"):
        st.markdown(prompt_from_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 AutoBOT sedang berpikir dan mencari informasi..."):

            messages_for_agent = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    messages_for_agent.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages_for_agent.append(AIMessage(content=msg["content"]))

            try:
                response = st.session_state.agent.invoke({"messages": messages_for_agent})

                if "messages" in response and len(response["messages"]) > 0:
                    answer = response["messages"][-1].content
                else:
                    answer = "Mohon maaf, terjadi kesalahan dalam pemrosesan jawaban."

            except Exception as e:
                if "401 Client Error" in str(e) and "Tavily" in str(e):
                    answer = "⚠️ ERROR: Gagal memanggil Tavily Search API. Kunci Tavily di Secrets mungkin salah."
                elif "API key not valid" in str(e) or "invalid_api_key" in str(e):
                    answer = "⚠️ ERROR: Kunci API Gemini tidak valid. Harap periksa Google AI API Key Anda."
                else:
                    answer = f"⚠️ Terjadi error yang tidak terduga saat menjalankan agen: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    st.rerun()