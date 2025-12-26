//Backend Run
cd C:\Users\s.muthukumarasamy\Downloads\RAG_Project_1\backend
.\.venv\Scripts\Activate
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

//Frontend run
npm run dev

//To check API in backend

python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); print('XAI_API_KEY =', os.getenv('XAI_API_KEY'))"



grok api
faiss
fastapi
pydantic
React JS
