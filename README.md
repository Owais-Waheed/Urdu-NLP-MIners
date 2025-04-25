# Urdu NLP Miners: Advanced Summarization & Sentiment Analysis for Urdu Text  
**Team Members**:  
Owais Waheed (`ow07611`) & Khubaib Mukaddam (`mk07218`)  

🚀 **Project Overview**  
Urdu NLP Miners addresses the underexplored domain of Urdu text analysis through state-of-the-art summarization and sentiment detection. Leveraging open-source LLMs like LLaMA, our solution automates analysis of news media and social content, supporting journalism and policy-making with metrics-driven insights.

📌 **Key Features**  
📰 _Context-Aware Summarization_ - Generate concise summaries using ROUGE/BLEU-optimized models  
🧠 _Sentiment Detection_ - Classify text sentiment with F1-score validated models  
🖋️ _Urdu Script Processing_ - Robust handling of diacritics (zabar/zer/pesh) and morphological variants  
📊 _Streamlit Visualization_ - Interactive web demo for result exploration  
🌐 _Low-Resource Optimization_ - Techniques adaptable to other resource-constrained languages  

🏗️ **Technology Stack**  
**NLP Core**: BERT, Gemma, MT5
**Processing**: UrduHack, SpaCy with custom Urdu rules  
**Evaluation**: ROUGE, BLEU, sklearn metrics  
**Deployment**: Streamlit, Google Colab  
**Infrastructure**: AWS/GCP for scalable processing , Fall back localhost  

📚 **Data Sources**  
We utilize curated Urdu datasets including:  
- [Urdu News Dataset (Kaggle)](https://www.kaggle.com/datasets/saurabhshahane/urdu-news-dataset)  
- [Urdu News Headlines (GitHub)](https://github.com/mwaseemrandhawa/Urdu-News-Headline-Dataset)  
- Open Data Pakistan's News Corpus  
- BBC Urdu/Dawn archives (fair-use compliant)  

📊 **Evaluation Metrics**  
| Task              | Metrics                          | Target Performance |
|-------------------|----------------------------------|--------------------|
| Summarization     | ROUGE-1/2, BLEU                 | 70%+ Recall        |
| Sentiment Analysis| F1-Score, Accuracy               | 80%+ Consistency   |
| Efficiency        |     |       |

🌐 **Deployment**  
- **Web Demo**: Interactive Streamlit interface  


⚠️ **Challenges & Solutions**  
| Challenge                        | Solution                          |
|----------------------------------|-----------------------------------|
| Diacritic Variations             | Rule-based normalization pipeline |
| Morphological Complexity         | Urdu-specific stemmer integration |
| Low-Resource Constraints         | LLM fine-tuning techniques        |


📜 **License**  
This project is released under [MIT License](LICENSE.md)
