import logging

logger = logging.getLogger("chronosense")
logger.setLevel(logging.DEBUG)

# Optional: add console output
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# | Level      | Meaning                          | Example                             |
# | ---------- | -------------------------------- | ----------------------------------- |
# | `DEBUG`    | Detailed internal info (for dev) | “Query embeddings generated.”       |
# | `INFO`     | General events (normal flow)     | “Dataset loaded successfully.”      |
# | `WARNING`  | Something odd, but not failing   | “Some rows had missing timestamps.” |
# | `ERROR`    | Operation failed                 | “Failed to create Chroma index.”    |
# | `CRITICAL` | System crash-level issues        | “RAG chain initialization failed!”  |

