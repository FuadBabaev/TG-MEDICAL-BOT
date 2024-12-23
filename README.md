# TG-MEDICAL-BOT

TG-MEDICAL-BOT is a Telegram-based medical assistant bot designed to provide helpful and concise medical advice. The bot uses a RAG approach by combining a vector database of medical documents with the Vicuna language model to generate responses to user queries.

---

## Key Features

- Retrieves context from a medical dataset to provide accurate and contextualized answers.
- Generates responses using the **Vicuna-7b-v1.5** language model.
- Handles queries related to first aid, symptoms, treatments, and general medical knowledge.
- Incorporates a user-friendly interface on Telegram.

---

## Dataset Used

The bot leverages data from the [MEDAL dataset](https://github.com/McGill-NLP/medal), a medical dataset containing rich contextual information on various medical topics. This dataset was preprocessed and embedded into a vector database (**Chroma**) to enable efficient retrieval during user interactions.

---

## How It Works

1. **Data Preparation**:

   - We added 1,000 documents from the MEDAL dataset to the **Chroma vector database**.
   - Each document includes a piece of medical text and associated metadata (e.g., diagnosis).

2. **Model Integration**:

   - Used the **Vicuna-7b-v1.5** model for text generation.
   - Embedded medical documents using **InstructorEmbedding**.

3. **Bot Setup**:

   - Built on the Telegram Bot API using **python-telegram-bot**.
   - Questions are processed by querying the vector database and generating responses with the LLM.

---

## Usage

### Try the Bot

- **Bot Link**: [@MyMedicalAssistantBot](https://t.me/MyMedicalAssistantBot)

> **Note**: The bot is not currently deployed on a live server. To test its functionality, please contact [@Babaev\_F](https://t.me/Babaev_F) to execute the script.

### Preview

To see how the bot works, check out the preview by the following link:

- **[Exported chat](https://fuadbabaev.github.io/TG-MEDICAL-BOT/images/messages.html)**

---

## Examples of Questions

Here are a few sample questions you can ask the bot:

1. "What are the symptoms of pneumonia?"
2. "How to stop severe bleeding?"
3. "What should I do if someone is choking?"
4. "How can I treat a minor burn?"
5. "What are the side effects of ibuprofen?"

---

## Repository Structure

```
TG-MEDICAL-BOT/
├── DB/                                     # Vector database with medical documents
├── images/                                 # Files to present the work
├── tg_medical_bot.py                       # Python script for the Telegram bot
├── LLM_RAG_Kiyako_Babaev.ipynb             # Jupyter notebook for development and testing
├── requirements.txt                        # Dependencies for the project
└── README.md                               
```

---

## Contact

For any questions or assistance, please contact [@Babaev\_F](https://t.me/Babaev_F).

---

