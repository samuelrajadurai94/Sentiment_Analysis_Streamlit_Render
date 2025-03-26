import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import tempfile
import os
from groq import Groq

API_URL = "https://sentiment-analysis-call-transcripts-c5dqzeme2.vercel.app/"


GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
client_groq = Groq(api_key = GROQ_API_KEY)

def transcribe_with_groq(audio_filepath):
    #client = Groq(api_key = GROQ_API_KEY)
    audio_file =open(audio_filepath,'rb')
    transcription = client_groq.audio.transcriptions.create(
        model='whisper-large-v3',file = audio_file,
        language='en',response_format="verbose_json")

    return transcription

# Custom CSS for styling
TITLE = 'Sentiment Analysis on Sales Call Audio & Transcripts'
st.markdown(
    f"""
    <div style='text-align: center; padding: 10px; border: 2px solid black; background-color: linear-gradient(90deg, #ff8c00, #ff0080);'>
        <span style='font-size: 30px; font-weight: bold; color: #FF5733; text-transform: uppercase;'>{TITLE}</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()
#st.write("Upload your call data files to analyze sentiment.")

uploaded_file = st.file_uploader("Upload your sales call Audio or transcript files to analyze sentiment.", type=["txt","mp3", "wav"])

if uploaded_file is not None:
    
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension in ["mp3", "wav"]: 
        with st.spinner("Transcribing audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(uploaded_file.read())  # Write the uploaded file to temp
                temp_audio_path = temp_audio.name  # Get file path
            
            # Transcribe the audio
            
            transcription = transcribe_with_groq(temp_audio_path)
        
            #segments =transcription['segments']
            segment_texts = [seg["text"] for seg in transcription.segments] 
            readable_transcript ="\n".join(segment_texts)
            st.subheader("Transcript:")
            #content = uploaded_file.read().decode('utf-8')
            st.text_area("Transcript Content", readable_transcript , height=300)   
            # with open("transcript3.txt", "w") as f:
            #     f.write("\n".join(segment_texts)) 
    elif file_extension == "txt":
            
            content = uploaded_file.read().decode('utf-8')
            st.subheader("Transcript:")
            st.text_area("Transcript Content", content, height=300)                      

    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            if file_extension == "txt":
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(API_URL, files=files)
            elif file_extension in ["mp3", "wav"]:
                files = {"file": (uploaded_file.name.split(".")[0]+'.txt', transcription.text)}
                response = requests.post(API_URL, files=files)

            #st.write(response)
            if response.status_code == 200:
                result = response.json()
                st.success("Analysis Complete!")
                #st.json(result)

                st.title("OVERALL SENTIMENT OF GIVEN TRANSCRIPT")  # Title for the text box

                # Render the content with desired styling
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 10px; border: 2px solid black; background-color: #f9f9f9;'>
                        <span style='font-size: 24px; font-weight: bold; color: red; text-transform: uppercase;'>{result['sentiment']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Display sentiment score
                st.subheader("Sentiment Scores Visualization:")
                scores = result['scores']
                best_sent=result['sentiment']
                best_score = scores[best_sent]
                
                col1, col2 = st.columns(2)
                
                #scores = result['scores']
                df = pd.DataFrame(list(scores.items()), columns=['Sentiment', 'Score']) 

                # Create bar plot
                fig = px.bar(df, x='Sentiment', y='Score', title="Sentiment Scores", 
                            labels={'Sentiment': 'Sentiment Type', 'Score': 'Score Value'},
                            color='Sentiment')
                fig.update_layout(showlegend=False,autosize=False,width=400,height=500,
                                  plot_bgcolor='lightgray',  # Light gray background for plot area
                                  paper_bgcolor='white',     # White background for the canvas
                                  font_color='black',        # Font color for readability
                                  xaxis=dict(showgrid=True, gridcolor='white'),  # White grid lines for x-axis
                                  yaxis=dict(showgrid=True, gridcolor='white'),
                                  margin=dict(l=50,r=50,b=50,t=50,pad=4))
                col1.plotly_chart(fig)
                fig = go.Figure()

                # Add the second indicator trace (Best Sentiment)

                fig = fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=best_score,
                    domain={'row': 1, 'column': 1},
                    title={'text': "Confidence Score",
                           'font':{'color':'GREEN','size':30}
                           },
                    number={'font': {'color': 'red'}}  # Set the value color to red
                ))
                

                # Update layout
                fig.update_layout(
                    template={'data': {'indicator': [{'mode': "number+delta+gauge"}]}},
                    autosize=False,width=400,height=500,
                    margin=dict(l=20,r=50,b=50,pad=4))

                col2.plotly_chart(fig)
                
            else:
                st.error("Error: Unable to process the file.")
