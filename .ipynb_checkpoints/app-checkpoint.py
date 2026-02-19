import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "vit_emotion_model.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD EMOTION MODEL
# =====================================================
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if "head.weight" in checkpoint:
        num_classes = checkpoint["head.weight"].shape[0]
    elif "classifier.weight" in checkpoint:
        num_classes = checkpoint["classifier.weight"].shape[0]
    else:
        raise RuntimeError("Classifier not found")

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =====================================================
# EMOTION LABELS
# =====================================================
CLASS_NAMES = ["Ahegao", "Angry", "Happy", "Neutral", "Sad", "Surprise"]

# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# FOOD DATABASE
# =====================================================
food_db = pd.DataFrame([
    # Sad
    [1, "Vegetable Soup", "Sad", 120, 3, 3, "Asian", 150],
    [2, "Chocolate", "Sad", 300, 25, 20, "Western", 250],
    [3, "Warm Milk", "Sad", 130, 12, 5, "Any", 120],
    [4, "Mashed Potatoes", "Sad", 210, 4, 6, "Western", 200],
    [5, "Oatmeal", "Sad", 160, 6, 4, "Any", 140],

    # Angry
    [6, "Yogurt", "Angry", 90, 4, 2, "Any", 100],
    [7, "Smoothie", "Angry", 180, 12, 2, "Any", 180],
    [8, "Green Tea", "Angry", 5, 0, 0, "Any", 80],
    [9, "Banana", "Angry", 105, 14, 0, "Any", 60],
    [10, "Salad Bowl", "Angry", 140, 5, 3, "Western", 170],

    # Happy
    [11, "Biryani", "Happy", 450, 5, 25, "Asian", 350],
    [12, "Burger", "Happy", 500, 6, 30, "Western", 400],
    [13, "Pizza", "Happy", 480, 8, 22, "Western", 450],
    [14, "Fried Chicken", "Happy", 520, 4, 28, "Western", 420],
    [15, "Pasta Alfredo", "Happy", 430, 6, 18, "Western", 380],

    # Neutral
    [16, "Fruit Salad", "Neutral", 80, 10, 1, "Any", 120],
    [17, "Grilled Fish", "Neutral", 200, 2, 8, "Asian", 300],
    [18, "Rice & Dal", "Neutral", 220, 4, 6, "Asian", 200],
    [19, "Chapati & Vegetables", "Neutral", 190, 3, 5, "Asian", 180],
    [20, "Boiled Eggs", "Neutral", 155, 1, 11, "Any", 150],

    # Surprise
    [26, "Ice Cream", "Surprise", 270, 20, 15, "Western", 200],
    [27, "Cupcakes", "Surprise", 320, 30, 18, "Western", 250],
    [28, "Pancakes", "Surprise", 350, 18, 12, "Western", 280],
    [29, "Milkshake", "Surprise", 400, 35, 20, "Any", 300],
    [30, "Donuts", "Surprise", 450, 28, 25, "Western", 320],

    # Ahegao / Excited (if used)
    [31, "Energy Drink", "Ahegao", 210, 27, 0, "Any", 180],
    [32, "Spicy Noodles", "Ahegao", 380, 6, 14, "Asian", 260],
    [33, "Hot Wings", "Ahegao", 420, 3, 26, "Western", 340],
    [34, "Cheese Fries", "Ahegao", 460, 4, 28, "Western", 350],
    [35, "BBQ Platter", "Ahegao", 550, 5, 35, "Western", 500],

    # Extra healthy options
    [36, "Lentil Soup", "Sad", 180, 3, 4, "Asian", 160],
    [37, "Steamed Vegetables", "Neutral", 90, 2, 1, "Any", 120],
    [38, "Brown Rice Bowl", "Neutral", 210, 3, 4, "Any", 200],
    [39, "Grilled Chicken Breast", "Happy", 260, 0, 6, "Any", 280],
    [40, "Quinoa Salad", "Fear", 230, 4, 7, "Western", 260],
], columns=["food_id","food","emotion","calories","sugar","fat","cuisine","price"])

# =====================================================
# DUMMY RATINGS (Collaborative Filtering)
# =====================================================
ratings_data = pd.DataFrame([
    [1,1,4],[1,2,5],[1,4,3],
    [2,1,5],[2,3,4],[2,5,5],
    [3,4,5],[3,6,4],[3,7,3],
    [4,5,4],[4,6,5],[4,8,3],
], columns=["user_id","food_id","rating"])

user_item_matrix = ratings_data.pivot_table(
    index="user_id",
    columns="food_id",
    values="rating"
).fillna(0)

# =====================================================
# HEALTH RULES
# =====================================================
health_rules = {
    "Diabetes": {"sugar_max": 10},
    "Heart": {"fat_max": 10},
    "Obesity": {"calories_max": 300}
}

# =====================================================
# RULE-BASED FILTER
# =====================================================
def rule_based_filter(df, user):
    for cond in user["health_conditions"]:
        rule = health_rules.get(cond, {})
        if "sugar_max" in rule:
            df = df[df["sugar"] <= rule["sugar_max"]]
        if "fat_max" in rule:
            df = df[df["fat"] <= rule["fat_max"]]
        if "calories_max" in rule:
            df = df[df["calories"] <= rule["calories_max"]]

    df = df[df["price"] <= user["budget"]]
    return df

# =====================================================
# GROUP-BASED COLLABORATIVE FILTERING
# =====================================================
def collaborative_recommend_group(user_id, emotion, candidate_food_ids, top_n=3):
    if user_id not in user_item_matrix.index:
        return []

    # Filter matrix by candidate foods (emotion group)
    valid_ids = [fid for fid in candidate_food_ids if fid in user_item_matrix.columns]

    if len(valid_ids) == 0:
        return []

    filtered_matrix = user_item_matrix[valid_ids]

    similarity = cosine_similarity(filtered_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=filtered_matrix.index,
        columns=filtered_matrix.index
    )

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:]

    weighted_scores = pd.Series(dtype=float)

    for sim_user, sim_score in similar_users.items():
        sim_user_ratings = filtered_matrix.loc[sim_user]
        weighted_scores = weighted_scores.add(
            sim_user_ratings * sim_score,
            fill_value=0
        )

    weighted_scores = weighted_scores.sort_values(ascending=False)

    return weighted_scores.index.tolist()[:top_n]

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Emotion-Aware Hybrid Recommender", layout="centered")
st.title("😊 Emotion-Aware Hybrid Food Recommendation")

uploaded_file = st.file_uploader("Upload Facial Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    detected_emotion = CLASS_NAMES[pred.item()]
    st.markdown(f"## 🎭 Detected Emotion: **{detected_emotion}**")
    st.markdown(f"Confidence: **{conf.item()*100:.2f}%**")

    # ---------------- USER INPUT ----------------
    st.markdown("### 👤 User Settings")
    selected_user_id = st.selectbox("Select Dummy User ID", user_item_matrix.index.tolist())
    budget = st.slider("Budget (PKR)", 100, 500, 300)
    preferred_cuisine = st.multiselect(
        "Preferred Cuisine",
        ["Asian", "Western", "Any"],
        default=["Asian"]
    )

    st.markdown("### ❤️ Health Conditions")
    health_conditions = []
    if st.checkbox("Diabetes"):
        health_conditions.append("Diabetes")
    if st.checkbox("Heart"):
        health_conditions.append("Heart")
    if st.checkbox("Obesity"):
        health_conditions.append("Obesity")

    user = {
        "emotion": detected_emotion,
        "health_conditions": health_conditions,
        "budget": budget,
        "preferred_cuisine": preferred_cuisine
    }

    # ---------------- STEP 1: Emotion Filter ----------------
    emotion_group = food_db[food_db["emotion"] == detected_emotion]

    # ---------------- STEP 2: Rule-Based Filter ----------------
    filtered_foods = rule_based_filter(emotion_group, user)

    if filtered_foods.empty:
        st.warning("No food matches your constraints.")
    else:
        candidate_ids = filtered_foods["food_id"].tolist()

        # ---------------- STEP 3: Collaborative Filtering within Group ----------------
        recommended_ids = collaborative_recommend_group(
            selected_user_id,
            detected_emotion,
            candidate_ids
        )

        final_recommendations = filtered_foods[
            filtered_foods["food_id"].isin(recommended_ids)
        ]

        st.markdown("### 🍽 Final Recommendations")

        if final_recommendations.empty:
            st.write("No collaborative matches found in this emotion group.")
        else:
            for _, row in final_recommendations.iterrows():
                st.write(
                    f"✅ **{row['food']}** | "
                    f"Cuisine: {row['cuisine']} | "
                    f"Price: PKR {row['price']}"
                )
