import os, requests, streamlit as st
API=os.getenv('API_URL','http://localhost:8000')
st.set_page_config(page_title='CivicShield',page_icon='🛡️'); st.title('🛡️ CivicShield'); st.caption('AI-assisted toxic comment screening for online communities.')
text=st.text_area('Enter a comment',height=150)
if st.button('Check comment',type='primary',use_container_width=True) and text.strip():
 try:
  st.session_state.result=requests.post(f'{API}/predict',json={'text':text},timeout=10).json(); r=st.session_state.result
  (st.error if r['label']=='toxic' else st.success)(f"{r['label'].title()} — {r['toxicity_probability']:.1%} toxicity probability")
 except requests.RequestException: st.error('API unavailable.')
if 'result' in st.session_state:
 st.write('Was this prediction correct?'); a,b=st.columns(2)
 if a.button('Yes'): requests.post(f"{API}/feedback/{st.session_state.result['prediction_id']}",json={'correct':True}); st.toast('Feedback saved')
 if b.button('No'): requests.post(f"{API}/feedback/{st.session_state.result['prediction_id']}",json={'correct':False}); st.toast('Feedback saved')
st.info('Limitations: historical English data can miss sarcasm, context, new abuse patterns, and other languages. Use for human review, never automatic bans.')
