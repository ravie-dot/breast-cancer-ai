from fastapi import APIRouter
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from routes.analysis import analysis_history

router = APIRouter(prefix="/api", tags=["history"])

@router.get("/history")
def get_history(limit: int = 50, skip: int = 0):
    return {
        "total": len(analysis_history),
        "items": analysis_history[skip:skip + limit]
    }

@router.delete("/history/{analysis_id}")
def delete_record(analysis_id: str):
    global analysis_history
    before = len(analysis_history)
    analysis_history = [h for h in analysis_history if h["id"] != analysis_id]
    if len(analysis_history) == before:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": analysis_id}

@router.delete("/history")
def clear_history():
    analysis_history.clear()
    return {"message": "History cleared"}
