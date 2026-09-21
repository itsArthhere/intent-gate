import streamlit as st
from intent_gate.data import ROOT
from intent_gate.inference import Router

st.set_page_config(page_title='Intent Gate',page_icon='🚦')
st.title('Intent Gate')
st.write('Route an English request to a CLINC150 intent, or defer when the model score is too low.')
st.caption('Local demo. No request is sent to an external model. Abstention is not a safety guarantee.')
if not (ROOT/'models/router.joblib').exists():
    st.info('Train the local router first: python -m intent_gate.train')
    st.stop()

@st.cache_resource
def load_router():
    return Router()

router=load_router()
st.caption(f"Model: {router.artifact['name']} · validation-locked threshold: {router.artifact['threshold']:.4f}")
text=st.text_input('Request',placeholder='What is my bank balance?',max_chars=2000)
if st.button('Route request',type='primary'):
    try:
        result=router.route(text)
        if result['decision']=='route':
            st.success(f"Intent: {result['intent']}")
        else:
            st.info('Abstain — ask for clarification or send to a human. No downstream action is executed.')
        st.json(result)
    except ValueError as exc:
        st.warning(str(exc))
with st.expander('How the policy was chosen'):
    st.write('Maximize correctly routed validation requests with at most 5% empirical false acceptance on the 100 OOS validation examples. The held-out test set does not set this threshold. Scores are not guaranteed probabilities of correctness.')
