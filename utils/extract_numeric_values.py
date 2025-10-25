from typing import List, Any

def parse_stock_string(content):
    """Parse the stock data string from Document content"""
    lines = content.strip().split('\n')
    data = {}
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Convert values appropriately
            if key == 'Date':
                data[key] = value  # Keep as string, convert to datetime later
            elif key in ['Volume']:
                try:
                    data[key] = int(float(value))  # Handle large numbers
                except ValueError:
                    data[key] = value
            else:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
    
    return data if data else None

def extract_numeric_values(docs, col: str='Close') -> List[Any]:
    """
    Extract numeric values from a document column, handling missing or malformed data.
    """
    numeric_values = []
    for doc in docs:
        # Assuming CSVLoader returns a dict row in page_content
        # page_content could be like: {'Date': '2023-01-01', 'Open': '100', 'Close': '105', ...}
        row = parse_stock_string(doc.page_content)
        # print(f"[DEBUG] Processing row: {row}")
        try:
            # Convert Close price to float
            numeric_values.append(float(row[col]))
        except Exception as e:
            print(f"[WARN] Skipping row due to conversion error: {e}")
    return numeric_values