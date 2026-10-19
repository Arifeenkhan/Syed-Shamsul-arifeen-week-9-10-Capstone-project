import os
import requests
import streamlit as st

API = os.getenv('API_URL', 'http://localhost:8000')

st.set_page_config(page_title='CivicShield', page_icon='🛡️')
st.title('🛡️ CivicShield')
st.caption('AI-assisted toxic comment screening for online communities.')

text = st.text_area('Enter a comment', height=150)

if st.button('Check comment', type='primary', use_container_width=True) and text.strip():
  try:
    st.session_state.result = requests.post(
        f'{API}/predict', json={'text': text}, timeout=10
    ).json()
    r = st.session_state.result

    # Safely handle keys using .get() to prevent crashes
    label = r.get('label', 'safe')
    prob = r.get('toxicity_probability', 0.0)

    (st.error if label == 'toxic' else st.success)(
        f'{label.title()} — {prob:.1%} toxicity probability'
    )
  except requests.RequestException:
    st.error('API unavailable.')

if 'result' in st.session_state:
  st.write('Was this prediction correct?')
  a, b = st.columns(2)
  if a.button('Yes'):
    requests.post(
        f"{API}/feedback/{st.session_state.result.get('prediction_id', 1)}",
        json={'correct': True},
    )
    st.toast('Feedback saved')
  if b.button('No'):
    requests.post(
        f"{API}/feedback/{st.session_state.result.get('prediction_id', 1)}",
        json={'correct': False},
    )
    st.toast('Feedback saved')

st.info(
    'Limitations: historical English data can miss sarcasm, context, new abuse'
    ' patterns, and other languages. Use for human review, never automatic'
    ' bans.'
)
