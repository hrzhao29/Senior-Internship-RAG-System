from langchain_community.document_loaders import WebBaseLoader
import bs4 
from pathlib import Path

# Only relevant text, no links or images
bs4_strainer = bs4.SoupStrainer(["p","h3"])
loader = WebBaseLoader(
    web_paths = ("https://en.wikipedia.org/wiki/One_Piece",),
    bs_kwargs = {"parse_only": bs4_strainer},
)

docs = loader.load()
text = "\n\n".join(doc.page_content for doc in docs)

out_path = Path("docs/onepiece.txt")
out_path.parent.mkdir(parents=True,exist_ok=True)
out_path.write_text(text,encoding="utf-8")

print(f"Saved {len(text)} characters to {out_path}")
