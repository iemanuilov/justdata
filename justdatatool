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
import mimetypes
import codecs
import binascii
import ast
import toml
from io import BytesIO
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, LargeBinary
from sqlalchemy import text
from sqlalchemy_utils import database_exists, create_database
from io import StringIO
from streamlit_tags import st_tags
from openai import OpenAI
from datasets import load_dataset
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler

# Load secrets from secrets.toml
secrets = toml.load('secrets.toml')

# Get database connection details from secrets
db_user = secrets['database']['user']
db_password = secrets['database']['password']
db_host = secrets['database']['host']
db_port = secrets['database']['port']
db_name = secrets['database']['name']

# Connect to PostgreSQL database
engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Create database if it doesn't exist
if not database_exists(engine.url):
    create_database(engine.url)

metadata = MetaData()

# Define tables
users = Table('users', metadata,
    Column('username', String, primary_key=True),
    Column('password', String)
)

annotations = Table('annotations', metadata,
    Column('id', Integer, primary_key=True),
    Column('dataset_name', Text),
    Column('dataset_url', Text),
    Column('tags', Text),
    Column('justification', Text),
    Column('file', LargeBinary),
    Column('file_name', Text),
    Column('username', Text)
)

tags = Table('tags', metadata,
    Column('id', Integer, primary_key=True),
    Column('tag', Text, unique=True)
)

# Create tables if they don't exist
metadata.create_all(engine)

# Function to verify login credentials
def check_credentials(username, password, connection):
    # Hash the entered password
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    print(f"Login hash: {password_hash}")  # Debug line
    
    with engine.connect() as connection:
        result = connection.execute(
            users.select().where(users.c.username == username).where(users.c.password == password_hash)
        ).fetchone()
    
    return result is not None

# Function to register a new user
def register_user(username, password, connection):
    # Hash the password before storing it
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    print(f"Register hash: {password_hash}")  # Debug line
    
    with engine.connect() as connection:
        result = connection.execute(
            users.select().where(users.c.username == username)
        ).fetchone()

    # If the username already exists, return False
    if result is not None:
        return False

    # Otherwise, insert the new user and return True
    with engine.connect() as connection:
        connection.execute(
            users.insert().values(username=username, password=password_hash)
        )
        connection.commit()  # Commit the changes
        
    return True

# Function to save annotation to database
def save_annotation(dataset_name, dataset_url, tags, justification, uploaded_file, username):
    
    # Read the uploaded file as bytes
    #file_bytes = uploaded_file.read() if uploaded_file is not None else None
    #file_name = uploaded_file.name if uploaded_file is not None else None  # Save the file name
    #c.execute("INSERT INTO annotations (dataset_name, dataset_url, tags, justification, file, file_name, username) VALUES (?, ?, ?, ?, ?, ?, ?)",
    #          (dataset_name, dataset_url, tags, justification, file_bytes, file_name, username))
    #conn.commit()

    # Create a connection
    conn = engine.connect()

    # Read the uploaded file as bytes
    file_bytes = uploaded_file.read() if uploaded_file is not None else None
    file_name = uploaded_file.name if uploaded_file is not None else None  # Save the file name

    print(f"Data to insert: {dataset_name}, {dataset_url}, {tags}, {justification}, {file_bytes}, {file_name}, {username}")

    # Execute a SQL statement
    #conn.execute(text("INSERT INTO annotations (dataset_name, dataset_url, tags, justification, file, file_name, username) VALUES (:dataset_name, :dataset_url, :tags, :justification, :file_bytes, :file_name, :username)"),
          #{"dataset_name": dataset_name, "dataset_url": dataset_url, "tags": tags, "justification": justification, "file_bytes": file_bytes, "file_name": file_name, "username": username})

    try:
        # Execute a SQL statement
        conn.execute(text("INSERT INTO annotations (dataset_name, dataset_url, tags, justification, file, file_name, username) VALUES (:dataset_name, :dataset_url, :tags, :justification, :file_bytes, :file_name, :username)"),
              {"dataset_name": dataset_name, "dataset_url": dataset_url, "tags": tags, "justification": justification, "file_bytes": file_bytes, "file_name": file_name, "username": username})
        conn.execute(text("COMMIT"))
        print("Data inserted successfully")
    except Exception as e:
        print(f"Error inserting data: {e}")

    # Close the connection
    conn.close()

# Function to get annotations for a specific user
def get_annotations(username):
    #c.execute(f"SELECT * FROM annotations WHERE username='{username}'")
    #return c.fetchall()

    # Create a connection
    conn = engine.connect()

    # Execute a SQL statement
    result = conn.execute(text("SELECT * FROM annotations WHERE username=:username"), {"username": username})

    # Fetch all rows from the result of the SQL statement
    records = result.fetchall()

    # Close the connection
    conn.close()

    return records

# Function to export annotations to a ZIP file
def export_annotations(username):
    
    # Fetch annotations for the logged-in user
    #c.execute("SELECT * FROM annotations")
    #annotations = get_annotations(username)
    
    # Create directories if they don't exist
    #os.makedirs(f'annotations/{username}', exist_ok=True)
    #os.makedirs('exports', exist_ok=True)

    # Create a connection
    conn = engine.connect()

    # Fetch annotations for the logged-in user
    result = conn.execute(text(f"SELECT * FROM annotations WHERE username = :username"), {"username": username})
    annotations = result.fetchall()

    # Close the connection
    #conn.close()

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
                # Create a BytesIO stream from the binary data
                file_bytes_encoded = file_bytes.encode('utf-8')
                file_stream = BytesIO(file_bytes_encoded)

                with open(f'annotations/{username}/annotation{id}_{file_name}', 'wb') as f:
                    # Read binary data from the stream
                    f.write(file_stream.read())
        else:
            print(f"Unexpected number of values in annotation: {annotation}")

    # Create a Zip file
    #zip_file_name = f'{username}_annotations.zip'
    #zip_file_path = f'exports/{zip_file_name}'
    #with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    #    for foldername, subfolders, filenames in os.walk(f'annotations/{username}'):
    #        for filename in filenames:
    #            # Add file to the ZIP file and maintain its folder structure
    #            zipf.write(os.path.join(foldername, filename), 
    #                    os.path.relpath(os.path.join(foldername, filename), f'annotations/{username}'))
        # Define the directory where the files will be saved
    directory = f'annotations/{username}'

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Execute a SELECT query to get the file data
    result = conn.execute(text("SELECT file, file_name FROM annotations WHERE username = :username"), {"username": username})

    # Fetch all rows from the result
    rows = result.fetchall()

    for row in rows:
        file_bytes, file_name = row

        # Check if file_bytes is a bytes object
        if isinstance(file_bytes, bytes):
            # Write the file data to a file
            with open(os.path.join(directory, file_name), 'wb') as f:
                f.write(file_bytes)
        else:
            print(f"file_bytes is not a bytes object: {file_bytes}")

    # Create a Zip file
    zip_file_name = f'{username}_annotations.zip'
    zip_file_path = f'exports/{zip_file_name}'

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for foldername, subfolders, filenames in os.walk(directory):
            for filename in filenames:
                # Add file to the ZIP file and maintain its folder structure
                zipf.write(os.path.join(foldername, filename), 
                        os.path.relpath(os.path.join(foldername, filename), directory))

    # Add a download button for the Zip file
    with open(zip_file_path, 'rb') as f:
        zip_bytes = f.read()
        b64 = base64.b64encode(zip_bytes).decode()  # Convert file to base64 encoding
        download_link = f'<a href="data:file/octet-stream;base64,{b64}" download="{zip_file_name}" style="display: inline-block; padding: 10px 20px; color: white; background-color: #1e88e5; text-decoration: none; border-radius: 5px;">Download</a>'
        st.markdown(download_link, unsafe_allow_html=True)

    # Remove the 'annotations' directory
    shutil.rmtree('annotations')

    conn.close()

def create_download_link(annotation_id):
    #if isinstance(row['File'], bytes):
    #    b64 = base64.b64encode(row['File']).decode()  # Convert file to base64 encoding
    #else:
    #    file_data = row['File'].encode()  # Convert string to bytes
    #    b64 = base64.b64encode(file_data).decode()  # Convert file to base64 encoding
    #return f'<a href="data:file/octet-stream;base64,{b64}" download="{row["File Name"]}">Download</a>'

    # Check if the file data is in bytes format
    # Check if the file data is in bytes format
    # Fetch file data from the database
    # Ensure annotation_id is a single value, not a Series
    # Ensure annotation_id is a single value, not a Series
    if isinstance(annotation_id, pd.Series):
        annotation_id = annotation_id.iloc[0]
    
    with engine.connect() as connection:
        result = connection.execute(text("SELECT file, file_name FROM annotations WHERE id = :id"), {'id': annotation_id})
        row = result.fetchone()._asdict()

    # Check if the file data is a string and convert it to bytes
    if isinstance(row['file'], str):
        row['file'] = row['file'].encode()

    # Check if the file data is in bytes format
    if isinstance(row['file'], bytes):
        # Convert file data to base64 encoding
        b64 = base64.b64encode(row['file']).decode()

        # Determine the MIME type of the file
        mime_type, _ = mimetypes.guess_type(row['file_name'])
        mime_type = mime_type if mime_type else 'application/octet-stream'

        # Create the download link
        download_link = f'<a href="data:{mime_type};base64,{b64}" download="{row["file_name"]}" style="display: inline-block; padding: 10px 20px; color: white; background-color: #1e88e5; text-decoration: none; border-radius: 5px;">Download</a>'
        return download_link
    else:
        print(f"File data is not in bytes format. It's of type {type(row['file'])}.")
        return None

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
                with engine.connect() as connection:
                    result = check_credentials(username, password, connection)

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
        if "just_registered" not in st.session_state:
            st.session_state["just_registered"] = False

        new_username = st.text_input("Username", value=st.session_state["new_username"])
        new_password = st.text_input("Password", type='password', value=st.session_state["new_password"])

        if st.button("Register"):
            with engine.connect() as connection:
                result = register_user(new_username, new_password, connection)
            
            if result:
                st.success("Registered successfully. You can now login.")
                st.session_state["new_username"] = new_username
                st.session_state["new_password"] = new_password
                st.session_state["just_registered"] = True  # Set the flag to True after registering
            else:
                st.warning("This username is already registered. Please choose a different username.")

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
    with engine.connect() as connection:
        # Fetch predefined tags from database
        result = connection.execute(text("SELECT tag FROM tags"))
        predefined_tags = [row[0] for row in result.fetchall()]

    if action == "Manage Tags":
        st.subheader("🤓Manage your tags")
        
        # Add tag
        new_tag = st.text_input("Add a new tag")
        if st.button("Add Tag"):
            if new_tag:
                with engine.connect() as connection:
                    connection.execute(text("INSERT INTO tags (tag) VALUES (:new_tag) ON CONFLICT (tag) DO NOTHING"), {"new_tag": new_tag})
                    connection.commit()
                    st.success("Tag added successfully!")
                    # Fetch updated list of tags
                    result = connection.execute(text("SELECT tag FROM tags"))
                    predefined_tags = [row[0] for row in result.fetchall()]
            else:
                st.error("Please enter a tag.")

        # Delete tag
        tag_to_delete = st.selectbox("Select a tag from the list to delete", predefined_tags)
        if st.button("Delete Tag"):
            with engine.connect() as connection:
                connection.execute(text("DELETE FROM tags WHERE tag = :tag_to_delete"), {"tag_to_delete": tag_to_delete})
                connection.commit()
                st.success("Tag deleted successfully!")
                # Fetch updated list of tags
                result = connection.execute(text("SELECT tag FROM tags"))
                predefined_tags = [row[0] for row in result.fetchall()]
        
        # Edit tag
        tag_to_edit = st.selectbox("Select a tag to edit it", predefined_tags)
        edited_tag = st.text_input("New name for this tag", value=tag_to_edit)
        if st.button("Save changes"):
            if edited_tag:
                with engine.connect() as connection:
                    connection.execute(text("UPDATE tags SET tag = :edited_tag WHERE tag = :tag_to_edit"), {"edited_tag": edited_tag, "tag_to_edit": tag_to_edit})
                    connection.commit()
                    st.success("Tag updated successfully!")
                    # Fetch updated list of tags
                    result = connection.execute(text("SELECT tag FROM tags"))
                    predefined_tags = [row[0] for row in result.fetchall()]
            else:
                st.error("Please select a tag.")

    elif action == "Annotate Dataset":
        
        #st.sidebar.header("⚙️JUST Chatbot Settings")
        st.sidebar.markdown("## ⚙️JUST Chatbot Settings <small style='color: gray;'>🏷️BETA</small>", unsafe_allow_html=True)
        st.sidebar.markdown("<small>Please enter your OpenAI API Key to unlock the JUST chatbot.</small>", unsafe_allow_html=True)
        with st.sidebar.form(key='chatbot_settings_form'):
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            submit_button = st.form_submit_button(label='Submit')

        st.sidebar.markdown("<small>Use the OpenAI API key provided in the documentation. This is an example of an API 🔑:<br>`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`</small>", unsafe_allow_html=True)

        if 'chatbot_active' not in st.session_state:
                st.session_state['chatbot_active'] = False
    
        # Fetch predefined tags from database again in case they were updated
        with engine.connect() as connection:
            # Fetch predefined tags from database
            result = connection.execute(text("SELECT tag FROM tags"))
            predefined_tags = [row[0] for row in result.fetchall()]

        st.subheader("✍️Annotate Dataset")
        dataset_name = st.text_input("Dataset Name", key='annotate_dataset_name')
        dataset_url = st.text_input("Dataset URL", key='annotate_dataset_url')
        tags = st.multiselect("Predefined tags", predefined_tags, key='annotate_tags')
        #user_tags = st.text_input("User-defined tags (separated by commas)", key='annotate_user_tags')
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
            df = pd.read_csv(dataset_url)

            if submit_button:
                st.session_state['chatbot_active'] = True
            
            if st.session_state['chatbot_active']:
                #st.subheader("💬JUST explore and chat with your dataset")
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
                        temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
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

                # Display annotations as a table
                #st.table(df)
                # Display annotations as a dataframe with a background gradient
                #st.dataframe(df.style.background_gradient(cmap='viridis'))
                # Display annotations as a dataframe with a background gradient

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
    st.sidebar.markdown("© 2024 Industry Commons Foundation")

    # Close database connection
    engine.connect().close()

if __name__ == '__main__':
    main()