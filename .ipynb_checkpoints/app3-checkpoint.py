import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Emotion-Aware Food Recommender", layout="wide")

# =====================================================
# CUSTOM BEAUTIFUL UI
# =====================================================
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #667eea, #764ba2);
}
.main {
    background-color: #f5f7fa;
    padding: 2rem;
    border-radius: 15px;
}
.recommend-card {
    background-color: grey;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
}
.rank-badge {
    font-size: 18px;
    font-weight: bold;
    color: #6C63FF;
}
.section-title {
    font-size: 28px;
    font-weight: bold;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "vit_emotion_model.pth.zip"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if "head.weight" in checkpoint:
        num_classes = checkpoint["head.weight"].shape[0]
    elif "classifier.weight" in checkpoint:
        num_classes = checkpoint["classifier.weight"].shape[0]
    else:
        raise RuntimeError("Classifier head not found")

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

CLASS_NAMES = ["Ahegao", "Angry", "Happy", "Neutral", "Sad", "Surprise"]

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
ratings = pd.DataFrame([
    [1,11,5],[1,12,4],
    [2,11,5],[2,13,5],
    [3,12,5],[3,13,4],
    [4,26,5],
], columns=["user_id","food_id","rating"])

user_item_matrix = ratings.pivot_table(
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
# RULE FILTER
# =====================================================
def rule_filter(df, user):
    df = df[df["price"] <= user["budget"]]
    for cond in user["health"]:
        rule = health_rules.get(cond, {})
        if "sugar_max" in rule:
            df = df[df["sugar"] <= rule["sugar_max"]]
        if "fat_max" in rule:
            df = df[df["fat"] <= rule["fat_max"]]
        if "calories_max" in rule:
            df = df[df["calories"] <= rule["calories_max"]]
    return df

# =====================================================
# COLLABORATIVE FILTERING
# =====================================================
def collaborative(user_id):
    if user_id not in user_item_matrix.index:
        return {}

    similarity = cosine_similarity(user_item_matrix)
    sim_df = pd.DataFrame(similarity,
                          index=user_item_matrix.index,
                          columns=user_item_matrix.index)

    similar_users = sim_df[user_id].sort_values(ascending=False)[1:]

    scores = {}

    for sim_user, sim_score in similar_users.items():
        for food_id in user_item_matrix.columns:
            rating = user_item_matrix.loc[sim_user, food_id]
            if rating > 0:
                scores[food_id] = scores.get(food_id, 0) + rating * sim_score

    return scores

# =====================================================
# FINAL WEIGHTED SCORING
# =====================================================
def compute_final_score(row, user, cf_scores):

    score = 0

    # Emotion weight
    if row["emotion"] == user["emotion"]:
        score += 5

    # Budget proximity
    price_diff = abs(user["budget"] - row["price"])
    score += max(0, 2 - (price_diff / user["budget"]))

    # Collaborative weight
    score += cf_scores.get(row["food_id"], 0)

    return score

# =====================================================
# UI
# =====================================================
st.markdown("<div class='section-title'>🍽 Emotion-Aware Hybrid Food Recommendation</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Facial Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image, use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)

    if torch.isnan(outputs).any():
        st.error("Model output error.")
    else:
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

        detected_emotion = CLASS_NAMES[pred.item()]

        with col2:
            st.markdown("### 🎭 Detected Emotion")
            st.markdown(f"""
            <div class="recommend-card">
                <h2>{detected_emotion}</h2>
                <p>Confidence: {conf.item()*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("## ⚙ Personal Settings")

        col3, col4 = st.columns(2)

        with col3:
            selected_user = st.selectbox("Select Dummy User", user_item_matrix.index.tolist())
            budget = st.slider("Budget (PKR)", 100, 600, 300)

        with col4:
            health = []
            if st.checkbox("Diabetes"):
                health.append("Diabetes")
            if st.checkbox("Heart"):
                health.append("Heart")
            if st.checkbox("Obesity"):
                health.append("Obesity")

        user = {
            "emotion": detected_emotion,
            "budget": budget,
            "health": health
        }

        emotion_foods = food_db[food_db["emotion"] == detected_emotion]
        filtered = rule_filter(emotion_foods, user)

        st.markdown("---")
        st.markdown("## 🏆 Top 5 Ranked Recommendations")

        if filtered.empty:
            st.warning("No food found.")
        else:
            cf_scores = collaborative(selected_user)

            filtered["final_score"] = filtered.apply(
                lambda row: compute_final_score(row, user, cf_scores),
                axis=1
            )

            filtered = filtered.sort_values(by="final_score", ascending=False)

            medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]

            for i, (_, row) in enumerate(filtered.head(5).iterrows()):
                st.markdown(f"""
                <div class="recommend-card">
                    <div class="rank-badge">{medals[i]} Rank {i+1}</div>
                    <h4>{row['food']}</h4>
                    <p>
                        💰 Price: {row['price']} PKR <br>
                        🔥 Calories: {row['calories']} <br>
                        ⭐ Final Score: {row['final_score']:.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
