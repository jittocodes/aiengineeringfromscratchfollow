from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("AI Sticky Notes")

NOTES_FILE = os.path.join(os.path.dirname(__file__), "notes.txt")

def ensure_file():
    if not os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "w") as f:
            f.write("")

@mcp.tool()
def add_note(note: str) -> str:
    """Add a note to the notes file

    :param note: The note to be added
    :return: Confirmation message indicating the note has been added.
    """
    ensure_file()
    with open(NOTES_FILE, "a") as f:
        f.write(note + "\n")
    return "Note added successfully."

# @mcp.tool() 
# def get_all_notes() -> str:
#     """Get all notes from the notes file"""
#     ensure_file()
#     with open(NOTES_FILE, "r") as f:
#         notes = f.read()
#     return notes.strip() if notes else "No notes found."
