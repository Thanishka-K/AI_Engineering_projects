from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os

app = FastAPI(title="Enterprise Ticket Routing Engine", version="1.0.0")

class TicketInput(BaseModel):
    ticket_id: str
    description: str

# Lazy load the classification pipeline securely
def load_pipeline():
    if not os.path.exists('ticket_router.pkl'):
        raise FileNotFoundError("Model binary missing. Please execute model_training.py first.")
    with open('ticket_router.pkl', 'rb') as f:
        return pickle.load(f)

try:
    router_pipeline = load_pipeline()
except Exception:
    router_pipeline = None

@app.post("/route-ticket/")
async def route_ticket(ticket: TicketInput):
    if router_pipeline is None:
        raise HTTPException(status_code=500, detail="Prediction pipeline uninitialized on host machine.")
    
    if not ticket.description.strip():
        raise HTTPException(status_code=400, detail="Ticket description cannot be empty.")

    # Execute text classification prediction
    predicted_dept = router_pipeline.predict([ticket.description])[0]
    probabilities = router_pipeline.predict_proba([ticket.description])[0]
    confidence = max(probabilities)

    return {
        "ticket_id": ticket.ticket_id,
        "assigned_department": predicted_dept,
        "routing_confidence": f"{confidence * 100:.2f}%",
        "status": "Routed Automatically"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
  
