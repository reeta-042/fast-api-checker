from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

# Load environment variable securely
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

if not pinecone_api_key:
    raise ValueError("⚠️ PINECONE_API_KEY is not set!")
if not pinecone_env:
    raise ValueError("⚠️ PINECONE_ENVIRONMENT is not set!")

# Initialize Pinecone and indexes
pc = Pinecone(api_key=pinecone_api_key)
drug_index = pc.Index("fake-product-drugs")
baby_index = pc.Index("fake-product-baby")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FastAPI app
app = FastAPI(
    title="Fake Product Checker API",
    description="API to verify whether a drug or baby product is real or fake using similarity matching.",
    version="1.0.0"
)


# =========================
# Pydantic input schemas
# =========================

class BabyProductInput(BaseModel):
    name: str
    brand_name: str
    price_in_naira: int
    platform: str
    product_type: str
    age_group: str
    package_description: str
    visible_expiriry_date: str


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


# =========================
# Classify functions
# =========================

def classify_product(user_text, index, threshold=0.8):
    vector = model.encode(user_text).tolist()
    result = index.query(vector=vector, top_k=5, include_metadata=True)

    if not result["matches"]:
        return "⚠️ No similar product found in the database."

    for match in result["matches"]:
        score = match["score"]
        text = match["metadata"]["text"]

        if score >= threshold:
            if "fake" in text.lower():
                reason = text.split("Reason:")[-1].strip() if "Reason:" in text else "No specific reason provided."
                return f"❌ Product is likely FAKE (score: {score:.2f})\nReason: {reason}\nI recommend you don't use this product. Please stay safe!"
            elif "real" in text.lower():
                reason = text.split("Reason:")[-1].strip() if "Reason:" in text else "No specific reason provided."
                return f"🎉 Product seems REAL (score: {score:.2f})\nReason: {reason}\nStay safe and shop wisely!"

    top_score = result["matches"][0]["score"]
    return f"⚠️ Product is unfamiliar or not similar enough (max score: {top_score:.2f}).\nSorry, I can't determine if this product is real or fake. Please verify manually or check with others. Stay safe!"


# =========================
# FastAPI endpoints
# =========================

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
    return {"result": classify_product(user_text, index=baby_index)}


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
    return {"result": classify_product(user_text, index=drug_index)}


#Starting main
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
