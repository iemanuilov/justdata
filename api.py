#JUST Data API
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to PostgreSQL database
try:
    conn = psycopg2.connect(
        dbname=st.secrets["postgres"]["database"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        sslmode=st.secrets["postgres"]["sslmode"],
        cursor_factory=RealDictCursor  # Use RealDictCursor to get dictionaries directly
    )
    c = conn.cursor()
except Exception as e:
    logger.error(f"Error connecting to the database: {e}")
    raise

# Pydantic models
class AnnotationCreate(BaseModel):
    dataset_id: int
    annotation: str

class AnnotationUpdate(BaseModel):
    annotation: str

# Root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "API is running"}

# CRUD endpoints
@app.get("/annotations")
def get_annotations():
    try:
        c.execute("SELECT id, dataset_url, dataset_name, tags, justification, file_name, username FROM annotations")
        annotations = c.fetchall()
        logger.info(f"Fetched annotations: {annotations}")
        return annotations
    except Exception as e:
        logger.error(f"Error fetching annotations: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/annotations")
def add_annotation(annotation: AnnotationCreate):
    try:
        c.execute(
            "INSERT INTO annotations (dataset_id, annotation) VALUES (%s, %s) RETURNING id",
            (annotation.dataset_id, annotation.annotation)
        )
        conn.commit()
        return {"id": c.fetchone()["id"]}
    except Exception as e:
        logger.error(f"Error adding annotation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.put("/annotations/{annotation_id}")
def update_annotation(annotation_id: int, annotation: AnnotationUpdate):
    try:
        c.execute(
            "UPDATE annotations SET annotation = %s WHERE id = %s",
            (annotation.annotation, annotation_id)
        )
        conn.commit()
        if c.rowcount == 0:
            raise HTTPException(status_code=404, detail="Annotation not found")
        return {"message": "Annotation updated successfully"}
    except Exception as e:
        logger.error(f"Error updating annotation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.delete("/annotations/{annotation_id}")
def delete_annotation(annotation_id: int):
    try:
        c.execute("DELETE FROM annotations WHERE id = %s", (annotation_id,))
        conn.commit()
        if c.rowcount == 0:
            raise HTTPException(status_code=404, detail="Annotation not found")
        return {"message": "Annotation deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting annotation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)