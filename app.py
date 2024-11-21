import streamlit as st
from openai import OpenAI
import time
import json
from datetime import datetime
from models import models
from utils.search_utils import SearchEngine
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import tiktoken
import nest_asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Apply nest_asyncio to allow asyncio in Streamlit
nest_asyncio.apply()

# Auto-save function defined at the top
def auto_save():
    """Auto-save the chat history and settings."""
    if st.session_state.settings.get('auto_save', False):
        try:
            with open('chat_backup.json', 'w') as f:
                json.dump({
                    'messages': st.session_state.messages,
                    'settings': st.session_state.settings,
                    'timestamp': datetime.now().isoformat()
                }, f)
        except Exception as e:
            st.warning(f"Auto-save failed: {str(e)}")

def get_client(premium):
    api_key = (
        os.getenv('OPENROUTER_PREMIUM_API_KEY')
        if premium
        else os.getenv('OPENROUTER_API_KEY')
    )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def stop_generation():
    st.session_state.generating = False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_with_retry(client, messages, model, temperature, max_tokens):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens
    )

# Initialize token counter
def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # Fallback to rough estimation if tiktoken fails
        return len(text.split()) * 1.3

# Initialize session state with new additions
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'generating' not in st.session_state:
    st.session_state.generating = False
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'theme': 'Dark',
        'temperature': 0.7,
        'max_tokens': 2000,
        'auto_save': False,
        'show_time': True,
        'notifications': True,
        'compact_view': False,
        'system_prompt': "You are an expert code generator. Generate clean, well-documented code with detailed explanations.",
        'search_enabled': True,
        'show_search_results': True,
        'search_results_count': 3,
        'context_window': 10
    }
if 'history_array' not in st.session_state:
    st.session_state.history_array = []
if 'session_info' not in st.session_state:
    st.session_state.session_info = {
        'start_time': datetime.now(),
        'message_count': 0,
        'total_tokens': 0,
        'errors': [],
        'last_activity': datetime.now()
    }
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {
        'response_times': [],
        'token_counts': [],
        'error_rates': {'total': 0, 'last_hour': 0},
        'successful_generations': 0
    }
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = SearchEngine(
        max_results=st.session_state.settings.get('search_results_count', 3)
    )

# Enhanced CSS with wider generation area
st.markdown("""
    <style>
    /* Main container */
    .main .block-container {
        padding: 2rem 1rem !important;  /* Reduced horizontal padding */
        max-width: 1600px !important;   /* Increased from 1400px */
        margin: 0 auto !important;
    }

    /* Chat messages area */
    .stChatMessageContainer {
        margin-bottom: 100px !important;
        width: 100% !important;
        max-width: 1500px !important;   /* Added max-width */
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Input styling - wider input */
    .stChatInput {
        max-width: 1400px !important;   /* Increased from 1200px */
        margin: 0 auto !important;
        padding: 0 1rem !important;     /* Reduced padding */
        width: 100% !important;
    }

    /* Stop button - adjusted position for wider chat */
    .stop-button {
        position: fixed !important;
        right: calc(50% - 480px) !important;  /* Adjusted from -380px */
        bottom: 19px !important;
        z-index: 1000 !important;
    }

    .stop-button button {
        background-color: transparent !important;
        color: #ff4b4b !important;
        border: 1px solid #ff4b4b !important;
        padding: 0.3rem 0.8rem !important;
        font-size: 0.8rem !important;
        border-radius: 0.3rem !important;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Timestamp styling */
    .timestamp {
        font-size: 0.7rem;
        color: #666;
        margin-top: 0.3rem;
    }

    /* Sidebar width */
    section[data-testid="stSidebar"] {
        width: 300px !important;        /* Reduced from 350px */
    }

    /* Ensure proper scrolling */
    [data-testid="stAppViewContainer"] > section:first-child {
        overflow-y: auto;
        height: calc(100vh - 100px);  /* Account for input height */
    }

    /* Dark/Light mode variables */
    [data-theme="dark"] {
        --background-color: #1e1e1e;
        --text-color: #ffffff;
    }
    
    [data-theme="light"] {
        --background-color: #ffffff;
        --text-color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model and Premium Settings
    st.markdown("### 🤖 Model Settings")
    premium = st.checkbox("🌟 Premium Mode", value=False)
    
    model = st.selectbox(
        "Model",
        models
    )
    
    # System Prompt Setting
    st.markdown("### 🔧 System Prompt")
    st.session_state.settings['system_prompt'] = st.text_area(
        "Customize System Prompt",
        value=st.session_state.settings['system_prompt']
    )
    
    # Generation Settings
    st.markdown("### 🎯 Generation Settings")
    st.session_state.settings['temperature'] = st.slider(
        "Temperature", 0.0, 1.0, 
        st.session_state.settings['temperature']
    )
    st.session_state.settings['max_tokens'] = st.number_input(
        "Max Tokens", 100, 4000, 
        st.session_state.settings['max_tokens']
    )
    
    # Display Settings
    st.markdown("### 🎨 Display Settings")
    st.session_state.settings['theme'] = st.radio(
        "Theme", ['Light', 'Dark'], 
        index=['Light', 'Dark'].index(st.session_state.settings['theme'])
    )
    
    # History Management
    st.markdown("### 📚 History Management")
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.history_array = []
        st.session_state.favorites = []
        st.rerun()
    
    # Context Window
    st.session_state.settings['context_window'] = st.sidebar.slider(
        "Context Window (previous messages)", 
        1, 20, 
        st.session_state.settings['context_window']
    )
    
    # Search History
    st.markdown("### 🔍 Search History")
    search_term = st.text_input("Search messages")
    if search_term:
        filtered_messages = [
            msg for msg in st.session_state.messages 
            if search_term.lower() in msg['content'].lower()
        ]
        st.write(f"Found {len(filtered_messages)} messages:")
        for msg in filtered_messages:
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")

    # Export/Import
    st.markdown("### 💾 Export/Import")
    if st.button("Export Chat"):
        export_data = {
            'messages': st.session_state.messages,
            'settings': st.session_state.settings,
            'timestamp': datetime.now().isoformat()
        }
        st.download_button(
            "Download Chat",
            data=json.dumps(export_data, indent=2),
            file_name="chat_export.json",
            mime="application/json"
        )
    
    # Internet Search
    st.markdown("### 🌐 Internet Search")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.settings['search_enabled'] = st.checkbox(
            "Enable Search", 
            value=st.session_state.settings['search_enabled'],
            help="Enable internet search for better responses"
        )
    with col2:
        st.session_state.settings['show_search_results'] = st.checkbox(
            "Show Results", 
            value=st.session_state.settings['show_search_results'],
            help="Show search results in chat"
        )
    
    if st.session_state.settings['search_enabled']:
        st.session_state.settings['search_results_count'] = st.slider(
            "Number of results", 1, 5, 
            st.session_state.settings['search_results_count'],
            help="Maximum number of search results to use"
        )

# Main chat interface
st.title("💬 Code Generation Assistant")

# Main chat container for messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if st.session_state.settings['show_time'] and 'timestamp' in message:
                st.markdown(f"<span class='timestamp'>{message['timestamp']}</span>", 
                          unsafe_allow_html=True)

# Fixed input at bottom
prompt = st.chat_input(
    "Ask me to generate code...",
    key="chat_input",
    disabled=st.session_state.generating
)

# Stop button (only shows when generating)
if st.session_state.generating:
    st.markdown('<div class="stop-button">', unsafe_allow_html=True)
    if st.button("⏹️", key="stop_btn", help="Stop generation"):
        stop_generation()
    st.markdown('</div>', unsafe_allow_html=True)

async def process_with_search(prompt: str, messages: list, message_placeholder):
    """Process the prompt with internet search augmentation."""
    try:
        search_results = []
        if st.session_state.settings['search_enabled']:
            search_message = "🔎 Searching for relevant information..."
            message_placeholder.markdown(search_message)
            
            # Customize search query based on content type
            if "code" in prompt.lower() or "programming" in prompt.lower():
                search_query = f"{prompt} programming example code"
            elif "error" in prompt.lower():
                search_query = f"{prompt} solution fix stackoverflow"
            elif "how to" in prompt.lower():
                search_query = f"{prompt} tutorial guide"
            else:
                search_query = prompt
                
            # Perform search with progress indicator
            with st.spinner("Searching..."):
                search_results = await st.session_state.search_engine.async_search(search_query)
            
            if search_results:
                formatted_results = st.session_state.search_engine.format_search_results(search_results)
                if st.session_state.settings['show_search_results']:
                    message_placeholder.markdown(f"Found relevant information:\n\n{formatted_results}")
                
                # Enhance system prompt with search results
                search_context = (
                    f"\nRelevant information from research:\n{formatted_results}\n"
                    f"Use this information to provide an accurate, up-to-date response. "
                    f"Include relevant code examples and explanations when appropriate."
                )
                messages[0]["content"] = st.session_state.settings['system_prompt'] + search_context
            else:
                message_placeholder.markdown("No relevant information found. Generating response based on existing knowledge.")
        
        return messages
    except Exception as e:
        st.error(f"Search processing error: {str(e)}")
        return messages

async def handle_message_generation(prompt: str, client, model, messages, message_placeholder):
    """Handle the async message generation process"""
    start_time = time.time()
    full_response = ""
    
    # Process with search if enabled
    messages = await process_with_search(prompt, messages, message_placeholder)
    
    # Show generation starting
    message_placeholder.markdown("🤖 Generating response...")
    
    stream = generate_with_retry(client, messages, model, 
                               st.session_state.settings['temperature'],
                               st.session_state.settings['max_tokens'])
    
    for chunk in stream:
        if not st.session_state.generating:
            message_placeholder.markdown(full_response)
            return full_response, time.time() - start_time
            
        if hasattr(chunk.choices[0].delta, 'content'):
            content = chunk.choices[0].delta.content
            full_response += content
            message_placeholder.markdown(full_response + "▌")
        elif hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason:
            break  # Generation finished
    
    return full_response, time.time() - start_time

if prompt:
    # Stop any ongoing generation
    stop_generation()
    
    # Add user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": timestamp,
        "tags": [],
        "category": "code" if "```" in prompt else "question"
    }
    st.session_state.messages.append(user_message)
    st.session_state.history_array.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        st.session_state.generating = True
        
        try:
            client = get_client(premium)
            
            # Prepare messages
            messages = [{"role": "system", "content": st.session_state.settings['system_prompt']}]
            messages.extend(st.session_state.history_array[-st.session_state.settings['context_window']:])
            messages.append({"role": "user", "content": prompt})
            
            # Run async message generation
            full_response, generation_time = asyncio.run(
                handle_message_generation(
                    prompt, client, model, messages, message_placeholder
                )
            )
            
            # Final message update
            message_placeholder.markdown(full_response)
            
            # Add to history
            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.messages.append(assistant_message)
            st.session_state.history_array.append({"role": "assistant", "content": full_response})
            
            if st.session_state.settings['show_time']:
                st.info(f"⚡ Generation time: {generation_time:.2f} seconds")
            
            # Token counting and metrics
            tokens = count_tokens(full_response)
            st.session_state.session_info['total_tokens'] += tokens
            if st.session_state.session_info['total_tokens'] > 100000:
                st.warning("Approaching token limit for this session")
            
            # Performance metrics
            st.session_state.performance_metrics['response_times'].append(generation_time)
            st.session_state.performance_metrics['successful_generations'] += 1
            
            # Auto-save
            auto_save()
            
        except Exception as e:
            st.error(f"🚨 An error occurred: {str(e)}")
        finally:
            st.session_state.generating = False

st.markdown("---")
st.markdown("Made with ❤️ by Your Name")
