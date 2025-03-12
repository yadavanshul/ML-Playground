# AI-Powered EDA & Preprocessing Dashboard

An intelligent dashboard for exploratory data analysis and preprocessing with AI-driven insights and recommendations.

## Features

- **Dataset Management**: Upload custom datasets or select from predefined options
- **AI-Driven EDA**: Dynamic visualization with AI-generated insights
- **Preprocessing Pipeline**: Drag & drop interface for building preprocessing workflows
- **Multi-Agent AI System**: Specialized AI agents for different tasks
- **RAG & ChromaDB**: Enhanced AI decision-making with retrieval-augmented generation
- **Error Detection**: Automatic identification and guided correction of data issues

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```
   cd ai_eda_pipeline
   streamlit run main.py
   ```

## Project Structure

```
/ai_eda_pipeline
├── components/          # AI agents & ML functions
├── data/                # Predefined datasets
├── utils/               # Helper functions
├── main.py              # Streamlit App
├── chromadb_store/      # ChromaDB Embeddings
├── requirements.txt     # Dependencies
```

## Usage

1. Select or upload a dataset
2. Drag & drop visualizations to the dashboard
3. Customize graphs and request AI insights
4. Build preprocessing pipelines with the drag & drop interface
5. Get AI feedback on your preprocessing choices
6. Export the preprocessed dataset for modeling

## License

MIT 