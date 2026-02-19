# =====================================================
# AUTHENTICATION MODULE
# =====================================================
import streamlit as st
import pandas as pd
import os

FILE = "users.xlsx"

# =====================================================
# CREATE USER EXCEL FILE
# =====================================================
def create_user_file():

    if not os.path.exists(FILE):

        df = pd.DataFrame(columns=[
        "Email","Password","Name","Age","Gender","City",
        "Health","Allergies","Diet","Calories",
        "Income","Budget","Cuisine"
        ])

        df.to_excel(FILE,index=False)

# =====================================================
# REGISTER USER
# =====================================================
def register_user():

    st.subheader("📝 User Registration")

    email = st.text_input("Email")
    password = st.text_input("Password",
                             type="password")

    name = st.text_input("Full Name")
    age = st.number_input("Age",18,80)

    gender = st.selectbox("Gender",
                          ["Male","Female","Other"])
    city = st.text_input("City")

    health = st.multiselect(
        "Health Conditions",
        ["Diabetes","Heart"]
    )

    allergies = st.multiselect(
        "Allergies",
        ["Peanut","Milk"]
    )

    diet = st.multiselect(
        "Dietary Restrictions",
        ["Low Sugar","Low Fat"]
    )

    calories = st.number_input(
        "Daily Calorie Limit"
    )

    income = st.selectbox(
        "Monthly Income",
        ["<30k","30-60k","60-100k"]
    )

    budget = st.number_input(
        "Food Budget",
        100,1000
    )

    cuisine = st.multiselect(
        "Favourite Cuisine",
        ["Pakistani","Chinese","Italian"]
    )

    if st.button("Register"):

        df = pd.read_excel(FILE)

        if email in df["Email"].values:
            st.error("User already exists!")

        else:
            new_user = pd.DataFrame([[
            str(email).strip().lower(),
            str(password).strip(),
            name,age,gender,city,
            str(health),str(allergies),
            str(diet),calories,
            income,budget,str(cuisine)]],
            columns=df.columns)

            df = pd.concat([df,new_user],
                           ignore_index=True)

            df.to_excel(FILE,index=False)

            st.success("Registration Successful!")

# =====================================================
# LOGIN USER
# =====================================================
def login_user():

    st.subheader("🔐 Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        # Read Excel as STRING
        df = pd.read_excel(FILE, dtype=str)

        email = email.strip().lower()
        password = password.strip()

        df["Email"] = df["Email"].str.strip().str.lower()
        df["Password"] = df["Password"].str.strip()

        user = df[
            (df["Email"] == email) &
            (df["Password"] == password)
        ]

        if not user.empty:
            st.session_state.logged_in = True
            st.session_state.user = user.iloc[0]
            st.success("Login Successful!")
        else:
            st.error("Invalid Credentials! Please Register.")

# =====================================================
# AUTH MENU
# =====================================================
def auth_menu():

    create_user_file()

    st.session_state.setdefault(
        "logged_in",False
    )

    menu = st.sidebar.selectbox(
        "Menu",
        ["Login","Register"]
    )

    if not st.session_state.logged_in:

        if menu=="Register":
            register_user()

        if menu=="Login":
            login_user()

        return False

    else:

        st.sidebar.success(
        f"Welcome {st.session_state.user['Name']}"
        )

        return True

# =====================================
# TEMP TEST UI (REMOVE LATER)
# =====================================
create_user_file()

menu = st.sidebar.selectbox(
    "Menu",
    ["Login","Register"]
)

if menu=="Register":
    register_user()

if menu=="Login":
    login_user()
