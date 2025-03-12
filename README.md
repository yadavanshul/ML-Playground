# Machine Learning Playground, Using AI

An interactive data science platform that leverages AI to make exploratory data analysis, data preprocessing, and machine learning more intuitive and powerful.

![ML Playground Screenshot](https://via.placeholder.com/800x400?text=ML+Playground+Screenshot)

## 🚀 Overview

Machine Learning Playground is an end-to-end data science platform that combines the power of AI with interactive visualizations to help you understand your data, preprocess it effectively, and build machine learning models. The platform uses AI to analyze datasets, recommend visualizations, provide insights, and guide you through the entire machine learning pipeline.

## ✨ Features

### Phase 1: Exploratory Data Analysis (Completed)
- 📊 **AI-powered visualization recommendations** based on dataset characteristics
- 📈 **Interactive dashboard** with customizable plots
- 💡 **Real-time AI insights** that explain patterns and anomalies
- 📝 **Detailed reasoning logs** that show how the AI thinks about your data
- 🎨 **Beautiful, responsive UI** with intuitive controls

### Phase 2: Data Preprocessing (In Progress)
- 🧹 **Automated data cleaning** with AI guidance
- 🔍 **Intelligent feature engineering** suggestions
- 📉 **Outlier detection and handling**
- 🧩 **Missing value imputation** with smart strategies
- 🔄 **Interactive preprocessing workflow** builder

### Phase 3: Machine Learning Pipeline (Planned)
- 🤖 **Automated model selection** based on data characteristics
- 📚 **Model training and evaluation** with performance metrics
- 🔮 **Model explainability** features
- 📊 **Feature importance visualization**
- 🔄 **Hyperparameter tuning** with AI guidance

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-playground.git
cd ml-playground
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Run the application:
```bash
cd ai_eda_pipeline
streamlit run main.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Configure your OpenAI API key:
   - Click on the "Configure OpenAI API Key" section in the sidebar
   - Enter your API key (starts with `sk-`)
   - Click "Save API Key"

4. Load a dataset:
   - Choose to upload your own CSV file or select from predefined datasets
   - Click "Load Dataset"

5. Explore the data:
   - Use the AI-recommended visualizations
   - Add plots to your dashboard
   - Get AI insights for each visualization

## 🔑 API Key Configuration

This project uses OpenAI's API for generating insights and recommendations. You have two options:

1. **Frontend Configuration (Recommended for personal use)**:
   - Enter your API key directly in the web interface
   - Your key is stored only in your browser session and never saved on any server

2. **Environment Variable (Alternative)**:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

If no API key is provided, the application will use simulated AI responses.

## 📁 Project Structure

```
ml-playground/
├── ai_eda_pipeline/           # Main application package
│   ├── components/            # Core components
│   │   ├── ai_agents.py       # AI agent implementations
│   │   └── preprocessing_workflow.py  # Preprocessing pipeline
│   ├── utils/                 # Utility functions
│   │   ├── data_utils.py      # Data handling utilities
│   │   └── visualization_utils.py  # Visualization functions
│   └── main.py                # Main Streamlit application
├── data/                      # Sample datasets
├── notebooks/                 # Jupyter notebooks for development
├── tests/                     # Test suite
├── .env.example               # Example environment variables
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔮 Future Plans

- **Phase 2 Completion**: Enhance data preprocessing capabilities with more advanced techniques
- **Phase 3 Implementation**: Complete the ML pipeline with model training and evaluation
- **Deployment Options**: Add export functionality for trained models
- **Collaboration Features**: Enable sharing and collaboration on analyses
- **Custom Visualization Builder**: Allow users to create custom visualizations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [OpenAI](https://openai.com/) for the AI capabilities
- [Pandas](https://pandas.pydata.org/) and [Scikit-learn](https://scikit-learn.org/) for data processing

---

Built with ❤️ by [Your Name] 