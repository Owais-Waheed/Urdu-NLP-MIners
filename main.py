import streamlit as st
import matplotlib.pyplot as plt
from newspaper import Article
import nltk
import io
from io import StringIO
import PyPDF2
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from langdetect import detect
import torch
# from docx import Document

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="اردو نیوز ایکسپریس",
    page_icon="📰"
)

# Title and description
st.title("اردو نیوز ایکسپریس")
st.write("اردو خبروں کا خلاصہ اور جذباتی تجزیہ")
st.write("##")

# Cache the model loading to improve performance

@st.cache_resource
def load_sentiment_model():
    model_path = "./urdusenti"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# @st.cache_resource
# def load_summarizer_model():
#     # Using mT5 model which supports Urdu
#     model_name = "google/mt5-small"
#     return pipeline("summarization", model=model_name)

# # Load the summarizer model
# summarizer = load_summarizer_model()
# classifier = load_sentiment_model()
# summarizer, classifier = 0,0

# # Function to extract text from docx with fallback
# def extract_text_from_docx(file):
#     doc = Document(file)
#     return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

# Function to extract text from various file types
def extract_text_from_file(file):
    file_type = file.name.split('.')[-1].lower()
    
    if file_type == 'txt':
        return StringIO(file.getvalue().decode('utf-8')).read()
    # elif file_type == 'docx':
    #     return extract_text_from_docx(file)
    elif file_type == 'pdf':
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        st.error("فائل کی قسم معاون نہیں ہے۔ برائے مہربانی TXT، DOCX یا PDF فائل اپلوڈ کریں۔")
        return ""


summary_model = "./gemma"
summarizer = AutoModelForCausalLM.from_pretrained(summary_model)
sum_tokenizer = AutoTokenizer.from_pretrained(summary_model)
classifier = load_sentiment_model()

def get_summary(text, max_chunk_words=400, max_new_tokens=150, model=None, tokenizer=None):
    import nltk
    nltk.download('punkt', quiet=True)

    # Split into sentences and form chunks
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) <= max_chunk_words:
            current_chunk.append(sentence)
            current_length += len(words)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Process each chunk with the Gemma model
    all_summaries = []
    for chunk in chunks:
        prompt = f"مندرجہ ذیل خبر کا خلاصہ لکھیں:\n\n{chunk}\n\nخلاصہ:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        summary_part = decoded.split("خلاصہ:")[-1].strip()
        all_summaries.append(summary_part)

    return " ".join(all_summaries)

# Function to analyze sentiment using LLM
def get_sentiment(text, classifier):
    # Load the sentiment analysis model
    
    # Process long texts in chunks and aggregate results
    max_token_length = 512
    
    # If text is short enough, analyze directly
    if len(text.split()) < max_token_length:
        result = classifier(text)
        label = result[0]['label']
        score = result[0]['score']
        
        # Map the result to Urdu sentiment labels
        sentiment_mapping = {
            '1 star': "انتہائی منفی",
            '2 stars': "منفی",
            '3 stars': "غیر جانبدار",
            '4 stars': "مثبت",
            '5 stars': "انتہائی مثبت"
        }
        
        return sentiment_mapping.get(label, "غیر جانبدار"), score
    
    # For longer texts, split and analyze in chunks
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_token_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Analyze each chunk
    overall_score = 0
    for chunk in chunks:
        result = classifier(chunk)
        # Convert rating to numeric score (1-5)
        rating = int(result[0]['label'].split()[0])
        overall_score += (rating - 1) / 4  # Normalize to 0-1 range
    
    # Average the scores
    if chunks:
        overall_score /= len(chunks)
    
    # Map to Urdu sentiment
    if overall_score < 0.3:
        return "منفی", overall_score
    elif overall_score > 0.7:
        return "مثبت", overall_score
    else:
        return "غیر جانبدار", overall_score

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["یو آر ایل", "فائل اپلوڈ", "متن داخل کریں"])

with tab1:
    with st.form("url_form", clear_on_submit=True):
        url_input = st.text_input("خبر کا یو آر ایل داخل کریں")
        url_submit = st.form_submit_button("تلاش کریں")
    
    if url_submit and url_input:
        try:
            with st.spinner("خبر لوڈ ہو رہی ہے..."):
                article = Article(url_input)
                article.download()
                article.parse()
                
                # Check if the article is in Urdu
                try:
                    language = detect(article.text[:100])
                    if language != 'ur':
                        st.warning("یہ خبر اردو میں نہیں ہے۔ نتائج درست نہیں ہو سکتے۔")
                except:
                    pass
                
                # Display article details
                st.subheader("خبر کی تفصیلات:")
                st.write(f"### {article.title}")
                
                if article.top_image:
                    st.image(article.top_image)
                
                with st.spinner("خلاصہ بنایا جا رہا ہے..."):
                    summary = get_summary(article.text, summarizer=summarizer, tokenizer=sum_tokenizer)
                    st.subheader("خبر کا خلاصہ:")
                    st.write(summary)
                
                with st.spinner("جذباتی تجزیہ کیا جا رہا ہے..."):
                    sentiment, score = get_sentiment(article.text, classifier=classifier)
                    st.subheader("جذباتی تجزیہ:")
                    st.write(f"یہ خبر **{sentiment}** ہے")
                    st.progress(score)
                
        except Exception as e:
            st.error(f"کوئی خرابی ہوئی ہے۔ درست یو آر ایل داخل کریں۔ خرابی: {str(e)}")

with tab2:
    uploaded_file = st.file_uploader("اردو خبر کی فائل اپلوڈ کریں", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        with st.spinner("فائل کو پڑھا جا رہا ہے..."):
            text = extract_text_from_file(uploaded_file)
            
            if text:
                with st.spinner("خلاصہ بنایا جا رہا ہے..."):
                    summary = get_summary(text, summarizer=summarizer)
                    st.subheader("دستاویز کا خلاصہ:")
                    st.write(summary)
                
                with st.spinner("جذباتی تجزیہ کیا جا رہا ہے..."):
                    sentiment, score = get_sentiment(text, classifier=classifier)
                    st.subheader("جذباتی تجزیہ:")
                    st.write(f"یہ دستاویز **{sentiment}** ہے")
                    st.progress(score)

with tab3:
    text_input = st.text_area("اردو متن داخل کریں", height=300)
    analyze_button = st.button("تجزیہ کریں")
    
    if analyze_button and text_input:
        with st.spinner("خلاصہ بنایا جا رہا ہے..."):
            summary = get_summary(text_input)
            st.subheader("متن کا خلاصہ:")
            st.write(summary)
        
        with st.spinner("جذباتی تجزیہ کیا جا رہا ہے..."):
            sentiment, score = get_sentiment(text_input, classifier=classifier)
            st.subheader("جذباتی تجزیہ:")
            st.write(f"یہ متن **{sentiment}** ہے")
            st.progress(score)

# Add a sidebar for model configuration
with st.sidebar:
    st.header("ماڈل کی ترتیبات")
    st.info("یہ ایپلیکیشن اردو نیوز کا خلاصہ اور جذباتی تجزیہ کرتی ہے ")
    st.write("**اہم ہدایات:**")
    st.write("1. پہلی بار چلانے پر، ماڈل ڈاؤن لوڈ کرنے میں چند منٹ لگ سکتے ہیں")
    st.write("2. بڑے متن کے تجزیے میں زیادہ وقت لگ سکتا ہے")
    # st.write("3. براہ راست اردو متن سب سے بہتر نتائج دیتا ہے")

# Footer
st.markdown("---")
st.caption("اردو نیوز ایکسپریس - خبروں کا خلاصہ اور جذباتی تجزیہ")