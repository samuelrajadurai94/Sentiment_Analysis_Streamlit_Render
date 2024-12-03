import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import json

API_URL = "https://sentiment-analysis-call-transcripts-c5dqzeme2.vercel.app/"

st.title("Sentiment Analysis on Sales Call Transcripts")
st.write("Upload your call transcript files to analyze sentiment.")

uploaded_file = st.file_uploader("Choose a transcript file", type=["txt"])

if uploaded_file is not None:
    st.subheader("Uploaded Transcript")
    content = uploaded_file.read().decode('utf-8')
    st.text_area("Transcript Content", content, height=300)

    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
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
                
                sentiments = list(scores.keys())
                values = list(scores.values())
                
                # Create bar plot
                fig = px.bar(x=sentiments, y=values, title="Sentiment Scores", 
                            labels={'Sentiment': 'Sentiment Type', 'Score': 'Score Value'},
                            color=sentiments)
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
