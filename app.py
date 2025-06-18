import os
import base64
import streamlit as st
from openai import OpenAI, error

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar UI
st.sidebar.title('Settings')
st.sidebar.write('Enter your OpenAI API key or set OPENAI_API_KEY in environment')
apikey_input = st.sidebar.text_input('API Key', type='password', value=st.session_state.api_key)
if st.sidebar.button('Save Key'):
    st.session_state.api_key = apikey_input

model = st.sidebar.selectbox('Model', ['gpt-4.1', 'gpt-4.1-mini'])
tool = st.sidebar.selectbox('Tool', ['Chat', 'Image', 'TTS'])

# Initialize OpenAI client
client = OpenAI(api_key=st.session_state.api_key)

# Helper functions

def generate_text(prompt: str, model: str) -> str:
    if not prompt.strip():
        raise ValueError('Prompt is empty')
    try:
        resp = client.responses.create(model=model, input=prompt)
        return resp.output_text
    except error.APIError as e:
        raise ValueError(str(e))


def generate_image(prompt: str, model: str) -> bytes:
    if not prompt.strip():
        raise ValueError('Prompt is empty')
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            tools=[{"type": "image_generation"}],
        )
        images = [out.result for out in resp.output if out.type == 'image_generation_call']
        if not images:
            raise ValueError('No image returned')
        return base64.b64decode(images[0])
    except error.APIError as e:
        raise ValueError(str(e))


def synthesize_tts(text: str, voice: str, model: str) -> bytes:
    if not text.strip():
        raise ValueError('Text is empty')
    try:
        resp = client.audio.speech.create(
            model=model,
            input=text,
            voice=voice
        )
        return resp.audio
    except error.APIError as e:
        raise ValueError(str(e))

# UI handlers
st.title('OpenAI Responses App')

if tool == 'Chat':
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    prompt = st.chat_input('Ask something')
    if prompt:
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.write(prompt)
        with st.spinner('Generating...'):
            try:
                response = generate_text(prompt, model)
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                with st.chat_message('assistant'):
                    st.write(response)
            except ValueError as e:
                st.error(str(e))

elif tool == 'Image':
    prompt = st.text_input('Image prompt')
    if st.button('Generate') and prompt:
        with st.spinner('Generating image...'):
            try:
                img_bytes = generate_image(prompt, model)
                st.image(img_bytes)
                st.download_button('Download', data=img_bytes, file_name='image.png')
            except ValueError as e:
                st.error(str(e))

elif tool == 'TTS':
    text = st.text_area('Text to speak')
    voice = st.selectbox('Voice', ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'])
    if st.button('Speak') and text:
        with st.spinner('Synthesizing...'):
            try:
                audio_bytes = synthesize_tts(text, voice, 'tts-1')
                st.audio(audio_bytes, format='audio/mp3')
                st.download_button('Download', data=audio_bytes, file_name='speech.mp3')
            except ValueError as e:
                st.error(str(e))
