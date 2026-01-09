import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

ssd_path = "/Volumes/PortableSSD/PDS_eugenie"
df = pd.read_csv(os.path.join(ssd_path, "table_database.csv"))

st.title("Image Height Exploration Tool")

st.title("Image Height Exploration Tool")

# Only allow selection of years from 2015 to 2020 (for testing)
years = [year for year in sorted(df["annee"].unique()) if 2015 <= year <= 2020]
selected_year = st.selectbox("Année:", years)

filtered_data = df[df["annee"] == selected_year]

st.write(f"{len(filtered_data)} images found")

# Height histogram
fig, ax = plt.subplots()
ax.hist(filtered_data["hauteur"], bins=30, color="skyblue", edgecolor="black")
ax.set_title(f"Height distribution (année = {selected_year})")
ax.set_xlabel("Height (pixels)")
ax.set_ylabel("Number of images")
st.pyplot(fig)