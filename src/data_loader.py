# data_loader.py

from pathlib import Path
from typing import List
from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document


def load_time_series_documents(data_dir: str) -> List[Document]:
    """
    Loads time-series datasets (CSV, Excel, JSON) from a directory into LangChain Document format.
    This loader is optimized for financial datasets such as stocks, crypto, and yield curves.
    
    Each row is treated as a "record" document for chunking → embedding workflows.
    """
    data_path = Path(data_dir).resolve()
    print(f"[INFO] Scanning data directory: {data_path}")

    documents = []

    # --------------------
    # Load CSV files
    # --------------------
    csv_files = list(data_path.rglob("*.csv"))
    print(f"[INFO] Found {len(csv_files)} CSV files.")

    for csv_fp in csv_files:
        print(f"[LOAD] CSV → {csv_fp.name}")
        try:
            loader = CSVLoader(
                file_path=str(csv_fp),
                csv_args={"delimiter": ",", "quotechar": '"'}
            )
            docs = loader.load()
            # Add dataset name metadata
            for d in docs:
                d.metadata["source"] = csv_fp.name
                # Convert structured row text back into key:value dict
                row_dict = {}
                for line in d.page_content.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        row_dict[key.strip()] = value.strip()

                # Extract clean date
                date_str = row_dict.get("Date", None)
                if date_str:
                    # Convert "2020-10-26 00:00:00-04:00" → "2020-10-26"
                    date_str = date_str.split(" ")[0]

                # Store clean metadata (string + numbers only)
                d.metadata["Date"] = date_str
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed loading CSV {csv_fp}: {e}")

    # --------------------
    # Load Excel files
    # --------------------
    excel_files = list(data_path.rglob("*.xlsx"))
    print(f"[INFO] Found {len(excel_files)} Excel files.")

    for xl_fp in excel_files:
        print(f"[LOAD] Excel → {xl_fp.name}")
        try:
            loader = UnstructuredExcelLoader(str(xl_fp))
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = xl_fp.name
                # Convert structured row text back into key:value dict
                row_dict = {}
                for line in d.page_content.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        row_dict[key.strip()] = value.strip()

                # Extract clean date
                date_str = row_dict.get("Date", None)
                if date_str:
                    # Convert "2020-10-26 00:00:00-04:00" → "2020-10-26"
                    date_str = date_str.split(" ")[0]

                # Store clean metadata (string + numbers only)
                d.metadata["Date"] = date_str
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed loading Excel {xl_fp}: {e}")

    # --------------------
    # Load JSON files
    # --------------------
    json_files = list(data_path.rglob("*.json"))
    print(f"[INFO] Found {len(json_files)} JSON files.")

    for json_fp in json_files:
        print(f"[LOAD] JSON → {json_fp.name}")
        try:
            loader = JSONLoader(
                file_path=str(json_fp),
                jq_schema=".[]",   # Flatten array records if needed
                text_content=False,
            )
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = json_fp.name
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed loading JSON {json_fp}: {e}")

    print(f"[SUCCESS] Loaded total {len(documents)} documents.")
    return documents


if __name__ == "__main__":
    # Debug run:
    docs = load_time_series_documents("data/")
    print(f"Example Document:\n{docs[0] if docs else 'No documents found.'}")
