# JUST Data Annotation
# A Streamlit app for annotating datasets and exporting annotations to a ZIP file
import streamlit as st
import sqlite3
import zipfile
import base64
import json
import os
import io
import requests
import shutil
import pandas as pd
import sqlalchemy
import hashlib
import psycopg2
import xml.etree.ElementTree as ET
from PyPDF2 import PdfReader
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from io import StringIO
from streamlit_tags import st_tags
from openai import OpenAI
from datasets import load_dataset
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from pydantic_settings import BaseSettings

# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname=st.secrets["postgres"]["database"],
    user=st.secrets["postgres"]["user"],
    password=st.secrets["postgres"]["password"],
    host=st.secrets["postgres"]["host"],
    port=st.secrets["postgres"]["port"],
    sslmode=st.secrets["postgres"]["sslmode"],
)

# Create a cursor object
c = conn.cursor()

# Create table for users if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY,
              password TEXT)''')
# Commit the transaction
conn.commit()

# Create table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS annotations
             (id SERIAL PRIMARY KEY,
              dataset_name TEXT,
              dataset_url TEXT,
              tags TEXT,
              justification TEXT,
              file BYTEA,
              file_name TEXT,
              username TEXT)''')

# Commit the transaction
conn.commit()

# Function to verify login credentials
def check_credentials(username, password, c):
    # Hash the entered password
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password_hash))
    if c.fetchone() is not None:
        return True
    return False
    
# Create table for predefined tags if it doesn't exist; we also ensure that each tag is unique by asking SQLite to ignore the INSERT command in case of duplicate tags
c.execute('''CREATE TABLE IF NOT EXISTS tags
             (id SERIAL PRIMARY KEY,
              tag TEXT UNIQUE)''')
# Commit the transaction
conn.commit()

# Function to register a new user
def register_user(username, password, conn, c):
    # Hash the password before storing it
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    print(f"Register hash: {password_hash}")  # Debug line
    
    c.execute("SELECT * FROM users WHERE username=%s", (username,))
    if c.fetchone() is not None:
        st.warning("Username already exists. Please choose a different username.")
        st.stop()
    
    else:
        try:
            c.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password_hash))
            conn.commit()
            st.success("Registered successfully. Please go to login.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Function to save annotation to database
def save_annotation(dataset_name, dataset_url, tags, justification, uploaded_file, username):
    
    # Read the uploaded file as bytes
    file_bytes = uploaded_file.read() if uploaded_file is not None else None
    file_name = uploaded_file.name if uploaded_file is not None else None  # Save the file name
    c.execute("INSERT INTO annotations (dataset_name, dataset_url, tags, justification, file, file_name, username) VALUES (%s, %s, %s, %s, %s, %s, %s)",
              (dataset_name, dataset_url, tags, justification, file_bytes, file_name, username))
    conn.commit()

# Function to get annotations for a specific user
def get_annotations(username):
    c.execute(f"SELECT * FROM annotations WHERE username='{username}'")
    return c.fetchall()

# Function to export annotations to a ZIP file
def export_annotations(username):
    
    # Fetch annotations for the logged-in user
    c.execute("SELECT * FROM annotations")
    annotations = get_annotations(username)
    
    # Create directories if they don't exist
    os.makedirs(f'annotations/{username}', exist_ok=True)
    os.makedirs('exports', exist_ok=True)

    for annotation in annotations:
        if len(annotation) == 8:
            id, dataset_name, dataset_url, tags, justification, file_bytes, file_name, username = annotation
            data = {
                'id': id,
                'dataset_url': dataset_url,
                'dataset_name': dataset_name,
                'tags': tags,
                'justification': justification,
                'username': username
            }
            # Save annotation data to a JSON file
            with open(f'annotations/{username}/annotation_{id}.json', 'w') as f:
                json.dump(data, f, indent=4)
            
            # Save the uploaded file to a file
            if file_bytes is not None:
                if isinstance(file_bytes, str):
                    file_bytes = file_bytes.encode()  # Convert string to bytes
                with open(f'annotations/{username}/annotation{id}_{file_name}', 'wb') as f:
                    f.write(file_bytes)
        else:
            print(f"Unexpected number of values in annotation: {annotation}")

    # Create a Zip file
    zip_file_name = f'{username}_annotations.zip'
    with zipfile.ZipFile(f'exports/{zip_file_name}', 'w') as zipf:
        for foldername, subfolders, filenames in os.walk(f'annotations/{username}'):
            for filename in filenames:
                # Add file to the ZIP file and maintain its folder structure
                zipf.write(os.path.join(foldername, filename), 
                           os.path.relpath(os.path.join(foldername, filename), f'annotations/{username}'))

    # Add a download button for the Zip file
    with open(f'exports/{zip_file_name}', 'rb') as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download=\'{zip_file_name}\'>Click to download {zip_file_name}</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Remove the 'annotations' directory
    shutil.rmtree('annotations')

def create_download_link(row):
    if row['File'] is not None:
        file_bytes = row['File']
        if isinstance(file_bytes, str):
            file_bytes = file_bytes.encode()  # Convert string to bytes
        b64 = base64.b64encode(file_bytes).decode()  # Convert file to base64 encoding
        href = f'<a href="data:file/octet-stream;base64,{b64}" download="{row["File Name"]}">Download</a>'
    else:
        href = 'No file'
    return href

# Function to load dataset from a remote repository
def load_dataset(dataset_url):
    response = requests.get(dataset_url)
    data = pd.read_csv(StringIO(response.text))
    return data

# Main app
def main():
    st.title("⚖️JUST Data Annotation Tool")
    username = None  # Define username at the beginning of the function

    menu = ["About", "Login", "Register"]
    choice = st.sidebar.selectbox(
        "👋Welcome to JUST Data Annotation",
        menu,
        placeholder="Start here..."
        )

    if choice == "About":
        st.subheader("🗞️About")
        st.markdown(
        """
        Data plays a central role in developing and evaluating machine learning models. Responsible AI issues often stem from dataset characteristics. For instance, inadequate representation of diverse groups can lead to performance disparities in models. Unexpected correlations or anomalies in training data can hinder generalization. Subjective labels and misconceptions of ground truth can mislead models. Documenting datasets fosters reflection and transparency, helping creators and consumers make informed decisions. Good data documentation is crucial for responsible AI.
        """
        )
    
    # Check if the user is logged in
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        app(st.session_state["username"])  # Display the main app when the user is logged in

        # Add a logout button in the sidebar
        if st.sidebar.button('Logout'):
            # Clear the session state
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.sidebar.success('Logged out successfully.')
            st.rerun()  # Refresh the page

    elif choice == "Login":
        # Get the session state
        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False
        if "username" not in st.session_state:
            st.session_state["username"] = ""
        if "password" not in st.session_state:
            st.session_state["password"] = ""
        if "choice" not in st.session_state:
            st.session_state["choice"] = "Login"  # Default choice

        if st.session_state["choice"] == "Login":
            st.subheader("Login to JUST Data Annotation Tool")

            username = st.text_input("Username", value=st.session_state["username"])
            password = st.text_input("Password", type='password', value=st.session_state["password"])

            if st.button("Login"):
                result = check_credentials(username, password, c)

                if result:
                    st.success("Logged in as {}".format(username))
                    st.session_state["username"] = username
                    st.session_state["password"] = password
                    st.session_state["logged_in"] = True
                    app(st.session_state["username"])  # Forward the user to the "Annotate Dataset" view
                else:
                    st.warning("Incorrect username or password")

    elif choice == "Register":
        st.subheader("Create a new user account")

        if "new_username" not in st.session_state:
            st.session_state["new_username"] = ""
        if "new_password" not in st.session_state:
            st.session_state["new_password"] = ""

        new_username = st.text_input("Username", value=st.session_state["new_username"])
        new_password = st.text_input("Password", type='password', value=st.session_state["new_password"])

        if st.button("Register"):
            register_user(new_username, new_password, conn, c)
            st.session_state["new_username"] = new_username
            st.session_state["new_password"] = new_password
            st.session_state["just_registered"] = True  # Set the flag to True after registering
            if "just_registered" in st.session_state and st.session_state["just_registered"]:
                st.session_state["just_registered"] = False  # Reset the flag after displaying the message

def app(username):
    # Sidebar
    # st.sidebar.image("images/logo.png", use_column_width=True)  # Add a logo at the top of the sidebar
    st.sidebar.title("📝JUST Data Annotation")  # Use a larger font for the header

    # Add some space before the selectbox
    st.sidebar.markdown(" ")

    action = st.sidebar.selectbox("Select an action", ["Annotate Dataset", "View Annotations", "Export Annotations", "Manage Tags"])

    # Add some space after the selectbox
    st.sidebar.markdown(" ")

    st.sidebar.markdown("---")  # Add another horizontal line for separation

    # Fetch predefined tags from database
    c.execute("SELECT tag FROM tags")
    predefined_tags = [row[0] for row in c.fetchall()]

    if action == "Manage Tags":
        st.subheader("🤓Manage your tags")
        
        #Add tag
        st.markdown("### ➕ Add Tag")
        new_tag = st.text_input("Add a new tag")
        if st.button("Add Tag"):
            if new_tag:
                c.execute("INSERT INTO tags (tag) VALUES (%s) ON CONFLICT (tag) DO NOTHING", (new_tag,))
                conn.commit()
                st.success("Tag added successfully!")
            else:
                st.error("Please enter a tag.")
        
        #Delete tag
        st.markdown("### ❌ Delete Tag")
        tag_to_delete = st.selectbox("Select a tag from the list to delete", predefined_tags)
        if st.button("Delete Tag"):
            c.execute("DELETE FROM tags WHERE tag = %s", (tag_to_delete,))
            conn.commit()
            st.success("Tag deleted successfully!")
        
        # Edit tag
        st.markdown("### ✒️ Edit Tag")
        tag_to_edit = st.selectbox("Select a tag to edit it", predefined_tags)
        edited_tag = st.text_input("Enter a new name for this tag", value=tag_to_edit)
        if st.button("Save changes"):
            if edited_tag:
                c.execute("UPDATE tags SET tag = %s WHERE tag = %s", (edited_tag, tag_to_edit))
                conn.commit()
                st.success("Tag updated successfully!")
            else:
                st.error("Please enter a tag.")

    elif action == "Annotate Dataset":
        st.sidebar.markdown("## ⚙️JUST Chatbot Settings <small style='color: gray;'>🏷️BETA</small>", unsafe_allow_html=True)
        st.sidebar.markdown("<small>Enter your OpenAI API Key to activate the JUST chatbot powered by GPT-4 Turbo.</small>", unsafe_allow_html=True)
        with st.sidebar.form(key='chatbot_settings_form'):
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            submit_button = st.form_submit_button(label='Submit')

        st.sidebar.markdown("<small>🔑 Use the OpenAI API key provided in the documentation.</small>", unsafe_allow_html=True)

        if 'chatbot_active' not in st.session_state:
                st.session_state['chatbot_active'] = False
    
        # Fetch predefined tags from database again in case they were updated
        c.execute("SELECT tag FROM tags")
        predefined_tags = [row[0] for row in c.fetchall()]

        st.subheader("✍️Annotate Dataset")
        dataset_name = st.text_input("Dataset Name", key='annotate_dataset_name')
        dataset_url = st.text_input("Dataset URL", key='annotate_dataset_url')
        tags = st.multiselect("Predefined tags", predefined_tags, key='annotate_tags')
        user_tags = st_tags(
            label='Enter your own tags',
            text='Press enter to add more',
            value=[],
            suggestions=predefined_tags,
            key='annotate_user_tags'
        )
        justification = st.text_area("Justification", key='annotate_justification')
        uploaded_file = st.file_uploader("Choose a file", key='file_uploader')
        
        if st.button("Save Annotation", key='annotate_save_annotation'):
            all_tags = tags + user_tags
            save_annotation(dataset_name, dataset_url, ", ".join(all_tags), justification, uploaded_file, username)
            st.success("Annotation saved successfully!")
        
        if dataset_url:
            # Get the file extension
            _, file_extension = os.path.splitext(dataset_url)

            # Load the file based on its extension
            if file_extension == '.csv':
                df = pd.read_csv(dataset_url)
            elif file_extension == '.xlsx':
                df = pd.read_excel(dataset_url)
            elif file_extension == '.json':
                df = pd.read_json(dataset_url)
            elif file_extension == '.pdf':
                response = requests.get(dataset_url)
                with open('temp.pdf', 'wb') as file:
                    file.write(response.content)
                with open('temp.pdf', 'rb') as file:
                    pdf = PdfReader(file)
                    text = ''
                    for page in range(len(pdf.pages)):
                        text += pdf.pages[page].extract_text()
                    df = pd.DataFrame([text], columns=['Text'])
                os.remove('temp.pdf')  # delete the temporary file
            elif file_extension == '.xml':
                response = requests.get(dataset_url)
                tree = ET.ElementTree(ET.fromstring(response.content))
                root = tree.getroot()
                data = [{child.tag: child.text for child in root.iter()}]
                df = pd.DataFrame(data)
            else:
                raise ValueError(f'Unsupported file type: {file_extension}')

            if submit_button:
                st.session_state['chatbot_active'] = True
            
            if st.session_state['chatbot_active']:
                st.markdown("### 💬JUST explore and chat with your dataset <small style='color: gray;'>🏷️BETA</small>", unsafe_allow_html=True)
                st.dataframe(df) # Let's visualise the dataset
            # Initialize conversation history
                if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
                    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

                # Display chat messages
                for msg in st.session_state.messages:
                    st.chat_message(msg["role"]).write(msg["content"])

                # User input
                if prompt := st.chat_input(placeholder="What is this data about?"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.chat_message("user").write(prompt)

                    # Check for OpenAI API key
                    if not openai_api_key:
                        st.info("Please add your OpenAI API key to continue.")
                        st.stop()

                    # Initialize LLM
                    llm = ChatOpenAI(
                        temperature=0, model="gpt-4-turbo", openai_api_key=openai_api_key, streaming=True
                    )
                
                                # Create a pandas dataframe agent
                    pandas_df_agent = create_pandas_dataframe_agent(
                        llm,
                        df,
                        verbose=True,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        handle_parsing_errors=True,
                    )

                    # Get LLM response
                    with st.chat_message("assistant"):
                        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.write(response)

    elif action == "View Annotations":
        st.subheader("📃View Annotations")
        if username is not None:
            annotations = get_annotations(username)

            if annotations:  # Check if any annotations were returned
                # Convert annotations to pandas DataFrame
                df = pd.DataFrame(annotations, columns=['ID', 'Dataset Name', 'Dataset URL', 'Tags', 'Justification', 'File', 'File Name', 'User'])
                
                # Reorder the columns
                df = df.reindex(columns=['ID', 'Dataset Name', 'Dataset URL', 'Tags', 'Justification', 'File', 'File Name', 'Download File', 'User'])

                # Create a new 'Download' column with download links
                df['Download File'] = df.apply(create_download_link, axis=1)

                # Create a new 'Dataset URL' column with hyperlinks
                df['Dataset URL'] = df['Dataset URL'].apply(lambda x: f'<a href="{x}" target="_blank">Link</a>')
                
                df.drop(columns=['ID', 'File'], inplace=True)  # Remove the 'File' column
                
                # Sort the DataFrame by 'Dataset Name'
                df.sort_values('Dataset Name', inplace=True)

                # Set CSS properties for th elements in dataframe
                th_props = [
                    ('font-size', '13px'),
                    ('text-align', 'center'),
                    ('font-weight', 'bold'),
                    ('color', '#6d6d6d'),
                    ('background-color', '#f7f7f9')
                ]

                # Set CSS properties for td elements in dataframe
                td_props = [
                    ('font-size', '13px')
                ]

                # Set table styles
                styles = [
                    dict(selector="th", props=th_props),
                    dict(selector="td", props=td_props)
                ]

                # Apply the styles to the DataFrame
                df_styled = df.style.background_gradient(cmap='viridis').set_table_styles(styles)
                st.markdown(df_styled.to_html(escape=False, index=False), unsafe_allow_html=True)

                # Specify the columns you want to include in the report
                columns_to_include = ['Tags', 'Justification']

                # Generate a ProfileReport object for the specified columns
                profile = ProfileReport(df[columns_to_include], title="Annotations Analytics", explorative=True,
                        samples=None,
                        correlations=None,
                        missing_diagrams=None,
                        duplicates=None,
                        interactions=None,
                )

                # Display the ProfileReport
                st.subheader("📈Annotations Stats")
                st_profile_report(profile)
                
            else:
                st.warning("No annotations found.")
        else:
           st.warning("Please log in to view annotations.")

    elif action == "Export Annotations":
        st.subheader("📤Export Annotations")
        annotations = get_annotations(username)
        if st.button("Export to ZIP"):
            export_annotations(username)
            st.success("Annotations exported to ZIP file!")

    # Add a footer
    #st.sidebar.markdown("© 2024 Industry Commons Foundation")
    st.sidebar.markdown("© 2024 [Industry Commons Foundation](https://www.industrycommons.net)")

    # Close database connection
    conn.close()

if __name__ == '__main__':
    main()