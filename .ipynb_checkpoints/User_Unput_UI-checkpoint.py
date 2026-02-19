st.markdown("## 🧍‍♂️ Step 1: User Profile (Demo Mode)")

budget_level = st.selectbox(
    "Budget Preference",
    ["Low", "Medium", "High"],
    index=1
)

preferred_cuisine = st.multiselect(
    "Preferred Cuisine",
    ["Asian", "Western", "Pakistani", "Italian", "Any"],
    default=dummy_user_profile["preferred_cuisine"]
)

health_conditions = st.multiselect(
    "Health Conditions",
    ["Diabetes", "Heart", "Obesity"],
    default=dummy_user_profile["health_conditions"]
)

taste = st.selectbox(
    "Preferred Taste",
    ["Spicy", "Sweet", "Normal"],
    index=0
)




