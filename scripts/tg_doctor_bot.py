import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers.string import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings
import torch

emb_model = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large", 
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# The DB folder already contains a preloaded vector database with medical documents.
# These documents have been processed and their embeddings computed using the `HuggingFaceInstructEmbeddings`.
# This vector database can be used immediately for retrieval without needing to re-add or reprocess the data.
# The documents include medical text content with associated metadata, stored in the persisted directory.

persist_directory = "DB"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=emb_model)
print("VectorDB reloaded with embedding function.")

# (Vicuna model)
model_id = "lmsys/vicuna-7b-v1.5"
print("Initializing Vicuna model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)
llm = HuggingFacePipeline(pipeline=pipe)
print("Vicuna model initialized successfully.")

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""<s>Your role is a medical assistant. Use the following context to answer the user's question:
    {context}
    USER: {question}
    ASSISTANT:"""
)

def format_docs(docs):
    print("Formatting retrieved documents:", docs)  # Debug
    return "\n\n".join(doc.page_content for doc in docs)

def debug_retrieve_and_format(query):
    print(f"Querying retriever with question: {query}")
    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents(query)
    print(f"Retrieved docs: {docs}")
    formatted_docs = format_docs(docs)
    print(f"Formatted context for prompt: {formatted_docs}")
    return formatted_docs

# RAG chain
chain = (
    {
        "context": lambda x: debug_retrieve_and_format(x["question"]),
        "question": lambda x: x["question"]
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# Function to trim message to the last complete sentence within the Telegram limit
def trim_message_to_limit(message, limit=4096):
    """
    Trims the message to fit within the character limit, ensuring it ends at the last complete sentence.

    Args:
        message (str): The message to be trimmed.
        limit (int): The maximum allowed length of the message (default 4096 for Telegram).

    Returns:
        str: The trimmed message with a clear indication of truncation if needed.
    """
    if len(message) <= limit:
        return message

    sentences = message.split(". ")  
    trimmed_message = ""
    for sentence in sentences:
        if len(trimmed_message) + len(sentence) + 2 > limit:  
            break
        trimmed_message += sentence + ". "

    return trimmed_message.strip() + "...\n[Response truncated]" if len(trimmed_message) < len(message) else trimmed_message


# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command."""
    print("Received /start command from user.")
    suggested_questions = [
        "What are the symptoms of pneumonia?",
        "How to treat SARS?",
        "What causes high fever?",
        "What are the side effects of antibiotics?",
        "How to prevent the flu?",
    ]
    suggestion_message = (
        "Hi! I am your medical assistant bot. You can ask any medical question or pick from the suggested questions below:\n\n"
        + "\n".join([f"{i+1}. {q}" for i, q in enumerate(suggested_questions)])
        + "\n\nOr type your custom question."
    )
    await update.message.reply_text(suggestion_message)

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user queries."""
    user_query = update.message.text
    try:
        print(f"Received user query: {user_query}")
        response = chain.invoke({"question": user_query})
        print(f"Pipeline result: {response}")
        
        # Extract only the text after "ASSISTANT:" and trim it
        assistant_index = response.find("ASSISTANT:")
        if assistant_index != -1:
            response = response[assistant_index + len("ASSISTANT:"):].strip()
        response = trim_message_to_limit(response)
        
        await update.message.reply_text(response)
    except Exception as e:
        print(f"Error during query processing: {e}")
        await update.message.reply_text(f"Sorry, an error occurred: {e}")


if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    print("Bot is running... Waiting for queries.")
    app.run_polling()
