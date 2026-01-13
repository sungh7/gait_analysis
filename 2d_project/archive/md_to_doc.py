import markdown
import sys

def md_to_word(md_file, doc_file):
    with open(md_file, 'r') as f:
        text = f.read()
    
    html = markdown.markdown(text)
    
    # Add simple styling for Word
    html_content = f"""
    <html>
    <head>
    <style>
    body {{ font-family: Arial, sans-serif; }}
    </style>
    </head>
    <body>
    {html}
    </body>
    </html>
    """
    
    with open(doc_file, 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    md_to_word('/data/gait/2d_project/RESEARCH_PAPER_DRAFT.md', '/data/gait/2d_project/Research_Paper.doc')
    print("Converted to Research_Paper.doc")
