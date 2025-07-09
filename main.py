from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import os

# ✅ Load environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

if not pinecone_api_key:
    raise ValueError("⚠️ PINECONE_API_KEY is not set!")
if not pinecone_env:
    raise ValueError("⚠️ PINECONE_ENVIRONMENT is not set!")

# ✅ Initialize Pinecone and indexes
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
drug_index = pc.Index("drug-product")
baby_index = pc.Index("baby-product")

# ✅ Lazy load the smallest model
@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

# ✅ Start FastAPI app
app = FastAPI(
    title="Fake Product Checker API",
    description="Verify if a drug or baby product is real or fake using similarity search.",
    version="1.0.0"
)

# ✅ Baby product schema
class BabyProductInput(BaseModel):
    name: str
    brand_name: str
    price_in_naira: int
    platform: str
    product_type: str
    age_group: str
    package_description: str
    visible_expiriry_date: str

# ✅ Drug product schema
class DrugProductInput(BaseModel):
    drug_name: str
    price: int
    dosage: str
    form: str
    brand_name: str
    medicine_type: str
    pack_size: str
    indications: str
    side_effects: str
    expiry_date_available: str
    platform: str
    nafdac_number_present: str
    package_description: str

# ✅ Shared classify function
def classify_product(user_text, index, threshold=0.8):
    model = get_model()
    vector = model.encode(user_text).tolist()
    result = index.query(vector=vector, top_k=5, include_metadata=True)

    if not result["matches"]:
        return "⚠️ No similar product found in the database."

    for match in result["matches"]:
        score = match["score"]
        text = match["metadata"]["text"]

        if score >= threshold:
            if "fake" in text.lower():
                reason = text.split("Reason:")[-1].strip() if "Reason:" in text else "No reason provided."
                return f"❌ Likely FAKE (score: {score:.2f})\nReason: {reason}"
            elif "real" in text.lower():
                reason = text.split("Reason:")[-1].strip() if "Reason:" in text else "No reason provided."
                return f"✅ Looks REAL (score: {score:.2f})\nReason: {reason}"

    top_score = result["matches"][0]["score"]
    return f"⚠️ Product is unfamiliar (score: {top_score:.2f}). Please verify manually."

# ✅ Baby endpoint
@app.post("/verify-baby-product")
def verify_baby_product(data: BabyProductInput):
    user_text = f"""
    Product: {data.name}
    Brand: {data.brand_name}
    Price: {data.price_in_naira} NGN
    Platform: {data.platform}
    Type: {data.product_type}
    Age Group: {data.age_group}
    Package: {data.package_description}
    Expiry Visible: {data.visible_expiriry_date}
    """
    return {"result": classify_product(user_text, baby_index)}

# ✅ Drug endpoint
@app.post("/verify-drug-product")
def verify_drug_product(data: DrugProductInput):
    user_text = f"""
    Drug Name: {data.drug_name}
    Price: {data.price} NGN
    Dosage: {data.dosage}
    Form: {data.form}
    Brand: {data.brand_name}
    Medicine Type: {data.medicine_type}
    Pack Size: {data.pack_size}
    Indications: {data.indications}
    Side Effects: {data.side_effects}
    Expiry Date Visible: {data.expiry_date_available}
    Platform: {data.platform}
    NAFDAC Number Present: {data.nafdac_number_present}
    Package Description: {data.package_description}
    """
    return {"result": classify_product(user_text, drug_index)}

# ✅ Run it (Railway/Render will handle port env)
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
