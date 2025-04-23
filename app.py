import os
import dotenv
import tempfile
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

dotenv.load_dotenv()

st.set_page_config(page_title="AI PDF Chatbot and Agent", page_icon="📄")

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None

st.title("🤖 AI PDF Chatbot and Agent")
st.subheader("Upload your PDF and start asking questions!")

with st.sidebar:
    groq_api_key = st.text_input(
        "Enter your Groq API Key:",
        type="password",
        help="Get your API key from https://console.groq.com/keys",
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    model = "llama3-70b-8192"
    st.write(f"Using model: {model}")

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file and groq_api_key and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name

                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                document_chunks = text_splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vectorstore = FAISS.from_documents(document_chunks, embeddings)

                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer",
                )

                llm = ChatGroq(
                    model_name=model, temperature=0.2, groq_api_key=groq_api_key
                )
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                    memory=memory,
                    return_source_documents=True,
                )

                os.unlink(temp_file_path)

                st.session_state.processing_status = "success"
                st.sidebar.success("PDF processed successfully!")

            except Exception as e:
                st.session_state.processing_status = "error"
                st.sidebar.error(f"Error processing PDF: {str(e)}")

if st.session_state.conversation:
    user_question = st.chat_input("Ask a question about your PDF...")

    if user_question:
        try:
            st.session_state.chat_history.append(HumanMessage(content=user_question))

            with st.spinner("Thinking..."):
                response = st.session_state.conversation.invoke(
                    {"question": user_question}
                )
                answer = response["answer"]
                source_docs = response.get("source_documents", [])

            st.session_state.chat_history.append(AIMessage(content=answer))

        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

else:
    st.info("Please upload a PDF document and process it to start the conversation.")
