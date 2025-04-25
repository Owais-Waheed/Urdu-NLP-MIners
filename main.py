import streamlit as st
import matplotlib.pyplot as plt
from newspaper import Article
# import nltk 
import io
from io import StringIO
import PyPDF2
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
# from langdetect import detect
import torch
# from llm_guard.input_scanners import PromptInjection
# from llm_guard.output_scanners import Toxicity
# from llm_guard import scan_output
# from docx import Document

# def validate_summary(summary, original_text):
#     scanners = [
#         # Profanity(threshold=0.8, languages=["ur", "en"]),
#         Toxicity(threshold=0.7, languages=["ur", "en"]),
#         # SensitiveTopics(languages=["ur", "en"]),
#         # Hallucination(reference=original_text),
#     ]

#     issues, _ = scan_output(summary, scanners)
#     return issues

# Download necessary NLTK data
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize
    

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
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cpu")
    return model, tokenizer

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
# summarizer = AutoModelForCausalLM.from_pretrained(summary_model).to("cpu")
sum_tokenizer = AutoTokenizer.from_pretrained(summary_model)
summarizer = AutoModelForCausalLM.from_pretrained(summary_model,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False).to("cpu")
classifier, class_tokenizer = load_sentiment_model()

def get_summary(text, max_chunk_words=400, max_new_tokens=150, model=None, tokenizer=None):
    

    # Split into sentences and form chunks
    # Manually split the text into sentences using simple punctuation rules
    sentences = text.split('۔')  # Split on Urdu full stops (۔)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

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

def get_sentiment(text, tokenizer, model, threshold=0.6):
    model.eval()
    max_token_length = 512

    # Urdu sentiment label mapping (index: label)
    urdu_labels = [
          # 0
        "منفی",         # 1
        "غیر جانبدار",  # 2
        "مثبت",         # 3
           # 4
    ]

    # Split long text into chunks
    # Manually split the text into sentences using simple punctuation rules
    sentences = text.split('۔')  # Split on Urdu full stops (۔)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

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

    sigmoid = torch.nn.Sigmoid()
    all_probs = []

    # Run model on each chunk and collect probabilities
    for chunk in chunks:
        encoding = tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encoding)
        probs = sigmoid(outputs.logits)
        all_probs.append(probs)

    # Average predictions
    avg_probs = torch.mean(torch.stack(all_probs), dim=0).flatten()

    # Apply threshold and map to Urdu labels
    predicted_labels = (avg_probs >= threshold).int().tolist()
    urdu_predictions = [urdu_labels[i] for i, pred in enumerate(predicted_labels) if pred == 1]
    scores = avg_probs.tolist()

    return urdu_predictions[0], scores


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
                    summary = get_summary(article.text, model=summarizer, tokenizer=sum_tokenizer)
                    # issues = validate_summary(summary, original_text=article.text)

                    # if issues:
                    #     st.warning("خلاصے میں کچھ ممکنہ مسائل پائے گئے:")
                    #     for issue in issues:
                    #         st.markdown(f"- ⚠️ {issue}")
                    # else:
                    #     st.success("خلاصہ محفوظ اور مناسب ہے۔")

                    st.subheader("خبر کا خلاصہ:")
                    st.write(summary)
                
                with st.spinner("جذباتی تجزیہ کیا جا رہا ہے..."):
                    sentiment, score = get_sentiment(article.text, model=classifier, tokenizer=class_tokenizer)
                    st.subheader("جذباتی تجزیہ:")
                    st.write(f"یہ خبر **{sentiment}** ہے")
                    # st.progress(score)
                
        except Exception as e:
            st.error(f"کوئی خرابی ہوئی ہے۔ درست یو آر ایل داخل کریں۔ خرابی: {str(e)}")

with tab2:
    uploaded_file = st.file_uploader("اردو خبر کی فائل اپلوڈ کریں", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        with st.spinner("فائل کو پڑھا جا رہا ہے..."):
            text = extract_text_from_file(uploaded_file)
            
            if text:
                with st.spinner("خلاصہ بنایا جا رہا ہے..."):
                    summary = get_summary(text, model=summarizer, tokenizer=sum_tokenizer)
                    # issues = validate_summary(summary, original_text=article.text)

                    # if issues:
                    #     st.warning("خلاصے میں کچھ ممکنہ مسائل پائے گئے:")
                    #     for issue in issues:
                    #         st.markdown(f"- ⚠️ {issue}")
                    # else:
                    #     st.success("خلاصہ محفوظ اور مناسب ہے۔")

                    st.subheader("دستاویز کا خلاصہ:")
                    st.write(summary)
                
                with st.spinner("جذباتی تجزیہ کیا جا رہا ہے..."):
                    sentiment, score = get_sentiment(text,model=classifier, tokenizer=class_tokenizer)
                    st.subheader("جذباتی تجزیہ:")
                    st.write(f"یہ دستاویز **{sentiment}** ہے")
                    # st.progress(score)

with tab3:
    text_input = st.text_area("اردو متن داخل کریں", height=300)
    analyze_button = st.button("تجزیہ کریں")
    
    if analyze_button and text_input:
        with st.spinner("خلاصہ بنایا جا رہا ہے..."):
            summary = get_summary(text_input, model=summarizer, tokenizer=sum_tokenizer)
            # issues = validate_summary(summary, original_text=article.text)

            # if issues:
            #     st.warning("خلاصے میں کچھ ممکنہ مسائل پائے گئے:")
            #     for issue in issues:
            #         st.markdown(f"- ⚠️ {issue}")
            # else:
            #     st.success("خلاصہ محفوظ اور مناسب ہے۔")

            st.subheader("متن کا خلاصہ:")
            st.write(summary)
        
        with st.spinner("جذباتی تجزیہ کیا جا رہا ہے..."):
            sentiment, score = get_sentiment(text_input, model=classifier, tokenizer=class_tokenizer)
            st.subheader("جذباتی تجزیہ:")
            st.write(f"یہ متن **{sentiment}** ہے")
            # st.progress(score)

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