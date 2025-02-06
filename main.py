from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import base64
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

app = FastAPI()

@app.post("/analyze-food")
async def analyze_food_image(file: UploadFile = File(...)):
    # Read the image from the uploaded file
    image = Image.open(BytesIO(await file.read()))
    
    # Convert image to PNG format and encode as base64
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

    # Initialize ChatOllama model
    chat_model = ChatOllama(model="llava:7b")

    # Define the prompt
    prompt = """
    Identify the food item in the given image and provide details in **only** this format:

    Food Image Name: [Exact name of the food item]
    Weight (grams): [Weight of the dish in grams]
    Calories: [Approximate calorie count per serving]
    Protein: [Amount of protein per serving in grams]
    Fat: [Amount of fat per serving in grams]

    Do **not** include any explanation or extra text.
    """

    # Create HumanMessage object with text and image URL
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]
    )

    # Invoke chat model with the message
    response = chat_model.invoke([message])

    # Parse response using regex
    match = re.search(r"(?:Food|Dish|Fish) Image Name:\s*(.*?)\s*\n\s*Weight \(grams\):\s*(\d+\-?\d*)\s*\n\s*Calories:\s*([\d\-]+(?:\s*[\w]+)?)\s*\n\s*Protein:\s*(\d+\.*\d*)\s*\n\s*Fat:\s*(\d+\.*\d*)", response.content, re.DOTALL | re.IGNORECASE)

    if match:
        food_name = match.group(1).strip().lower() 
        food_name = food_name.replace('fillets', '') 

        weight = match.group(2).strip()
        calories = match.group(3).strip()
        protein = match.group(4).strip()
        fat = match.group(5).strip()

        if "per serving" not in calories:
            calories += " per portion"

        return JSONResponse(content={
            "food_name": food_name.title(),
            "weight": weight,
            "calories": calories,
            "protein": protein,
            "fat": fat
        })

    return JSONResponse(content={"error": "Unable to analyze food image"}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
