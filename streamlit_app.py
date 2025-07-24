import streamlit as st
import pandas as pd
import numpy as np
import random
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import fastf1
from fastf1 import get_session
import fastf1.plotting
import seaborn as sns
import os
cache_dir = '/tmp/fastf1_cache'  
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)
import matplotlib.pyplot as plt

######SITE FORMATTING######
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;  /* white background */
        color: #ffffff;  /* Black text color for contrast */
    }
    .stApp {
        background-color: #ffffff; /* Background for entire app */
    }
    .css-1lcbz5j {
        color: #aa1316;  /* Change the color of streamlit header */
    }
     section[data-testid="stSidebar"] * {
        color: white !important;
    }
    .stButton>button {
        background-color: #ffffff; /* white button */
        color: red;
    }
    .stButton>button:hover {
        background-color: #aa1316;  /* Darker red on hover */
    }
    .stSidebar {
        background-color: #aa1316;  /* Sidebar color */
    }
    .stAppHeader {
        background-color: #aa1316;
    }
    </style>
    """,
    unsafe_allow_html=True
)

fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False,
                          color_scheme=None)

AA23_valid_names = ("Alex Albon", "Albon", "alex albon", "albon", "23")
AKA12_valid_names = ("Andrea Kimi Antonelli", "Antonelli", "Kimi Antonelli", "andrea kimi antonelli", "antonelli", "kimi antonelli", "12")
CL16_valid_names = ("Charles Leclerc", "Leclerc", "charles leclerc", "leclerc", "16")
CS55_valid_names = ("Carlos Sainz", "carlos sainz", "Sainz", "carlos Sainz", "Sainz", "sainz", "55", "Smooth Operator", "smooth operator")
EO31_valid_names = ("Esteban Ocon", "Ocon", "esteban ocon", "ocon", "31")
FA14_valid_names = ("Fernando Alonso", "Alonso", "The Rookie", "Fernando", "fernando alonso", "alonso", "fernando", "the rookie", "14")
FC43_valid_names = ("Franco Colapinto", "Colapinto", "franco colapinto", "colapinto", "43")
GB5_valid_names = ("Gabriel Bortoleto", "Bortoleto", "gabriel bortoleto", "bortoleto", "Gabi", "gabi")
GR63_valid_names = ("George Russell", "Russell", "George Russel", "Russel", "george russell", "russell", "george russel", "russel", "63")
IH6_valid_names = ("Isack Hadjar", "Hadjar", "isack hadjar", "hadjar", "6")
JD7_valid_names = ("Jack Doohan", "Doohan", "jack doohan", "doohan", "7")
LH44_valid_names = ("Lewis Hamilton", "Hamilton", "lewis hamilton", "hamilton", "44")
LL30_valid_names = ("Liam Lawson", "Lawson", "liam lawson", "lawson", "30")
LN4_valid_names = ("Lando Norris", "Lando", "Norris", "lando norris", "lando", "norris", "4")
LS18_valid_names = ("Lance Stroll", "Stroll", "Daddy's Money", "lance stroll", "stroll", "daddy's money", "18")
MV33_valid_names = ("Max", "Max Verstappen", "Verstappen", "max verstappen", "verstappen", "1", "33", "2024 World Champion", "2024 world champion", "du du du du", "DU DU DU DU")
NH27_valid_names = ("Nico Hulkenberg", "Hulkenberg", "nico hulkenberg", "hulkenberg", "27")
OB87_valid_names = ("Oliver Bearman", "oliver bearman", "Ollie Bearman", "ollie bearman", "Ollie in the Wallie", "ollie in the wallie", "87")
OP81_valid_names = ("Oscar Piastri", "oscar piastri", "Piastri", "piastri", "Pastry", "pastry", "Great Barrier Chief", "great barrier chief", "Wizard of Aus", "wizard of aus", "oscar", "Oscar", "81")
PG10_valid_names = ("Pierre Gasley", "pierre gasley", "Gasley", "gasley", "10", "Pierre Gasly", "pierre gasly", "gasly")
YT22_valid_names = ("Yuki Tsunoda", "Tsunoda", "yuki tsunoda", "tsunoda", "22")
valid_driver_names = set(AA23_valid_names + AKA12_valid_names + CL16_valid_names + CS55_valid_names + EO31_valid_names + FA14_valid_names + GB5_valid_names + GR63_valid_names + IH6_valid_names + JD7_valid_names + LH44_valid_names + LL30_valid_names + LN4_valid_names + LS18_valid_names + MV33_valid_names + NH27_valid_names + OB87_valid_names + OP81_valid_names + PG10_valid_names + YT22_valid_names)

driver_name_to_code = {
        **dict.fromkeys(AA23_valid_names, "ALB"),
        **dict.fromkeys(AKA12_valid_names, "ANT"),
        **dict.fromkeys(CL16_valid_names, "LEC"),
        **dict.fromkeys(CS55_valid_names, "SAI"),
        **dict.fromkeys(EO31_valid_names, "OCO"),
        **dict.fromkeys(FA14_valid_names, "ALO"),
        **dict.fromkeys(FC43_valid_names, "COL"),
        **dict.fromkeys(GB5_valid_names, "BOR"),
        **dict.fromkeys(GR63_valid_names, "RUS"),
        **dict.fromkeys(IH6_valid_names, "HAD"),
        **dict.fromkeys(JD7_valid_names, "DOO"),
        **dict.fromkeys(LH44_valid_names, "HAM"),
        **dict.fromkeys(LL30_valid_names, "LAW"),
        **dict.fromkeys(LN4_valid_names, "NOR"),
        **dict.fromkeys(LS18_valid_names, "STR"),
        **dict.fromkeys(MV33_valid_names, "VER"),
        **dict.fromkeys(NH27_valid_names, "HUL"),
        **dict.fromkeys(OB87_valid_names, "BEA"),
        **dict.fromkeys(OP81_valid_names, "PIA"),
        **dict.fromkeys(PG10_valid_names, "GAS"),
        **dict.fromkeys(YT22_valid_names, "TSU")
    }

#############
option = st.sidebar.radio("Pages",["Home", "Compare Drivers", "Whole Races"])

############
####HOME####
if option == "Home":
    st.title("Compare Drivers for Ryann TeeHee")
    st.image("https://static01.nyt.com/images/2019/05/22/sports/21lauda2/a4e1a1b48f694cde9da4cce0791832c9-articleLarge.jpg?quality=75&auto=webp&disable=upscale")

####HOME####
############

############
####COMPARE DRIVERS####
if option == "Compare Drivers":
    st.write("im working on it")
    year = st.text_input("Season (e.g. 2024)")
    race = st.text_input("Grand Prix (e.g. Silverstone)")
    format_selection = st.selectbox("Type of Race", ["Qualifying", "Race"])
    drivers_input = st.text_input("Drivers seperated by commas (e.g. Verstappen, Piastri)")
    
    if format_selection == "Qualifying":
        if st.button("generate"):
                if year and race and drivers_input:
                    try:
                        with st.spinner('please wait while your telemetry is loading'):
                            if drivers_input:
                                session = get_session(int(year), race, 'Q')
                                session.load()
                                driver_names = [name.strip() for name in drivers_input.split(",")]
                                driver_codes = []
                                for name in driver_names:
                                    code = driver_name_to_code.get(name)
                                    if code:
                                        driver_codes.append(code)
                                    else:
                                        st.warning(f"Driver Not Recognized: {name}")
                                if len(driver_codes) < 1:
                                    st.warning("no valid drivers found")
                                else:
                                    fig, ax = plt.subplots()

                                    st.title("Sector Times")
                                    st.caption("Times taken from the fastest lap regardless of when throughout qualifying the drivers fastest lap was. Track development could play a factor if one driver made it further than another.")
                                    # Create an empty list to hold each driver's data
                        def format_time(timedelta_obj):
                            total_seconds = timedelta_obj.total_seconds()
                            minutes = int(total_seconds // 60)
                            seconds = total_seconds % 60
                            return f"{minutes}:{seconds:06.3f}"  # ensures leading zeros and 3 decimal places

                        rows = []

                        for code in driver_codes:
                            lap_data = session.laps.pick_driver(code).pick_fastest()

                            # Format each sector time
                            sector1 = format_time(lap_data['Sector1Time'])
                            sector2 = format_time(lap_data['Sector2Time'])
                            sector3 = format_time(lap_data['Sector3Time'])
                            lap_time = format_time(lap_data['LapTime'])

                            rows.append({
                                "Driver": code,
                                "Sector 1": sector1,
                                "Sector 2": sector2,
                                "Sector 3": sector3,
                                "Lap Time": lap_time
                            })

                        df = pd.DataFrame(rows)
                        st.dataframe(df)

                        fig, ax = plt.subplots()

                        st.title("Fastest Lap Comparison")
                        st.caption("Times taken from the fastest lap regardless of when throughout qualifying the drivers fastest lap was. Track development could play a factor if one driver made it further than another.")
                        for code in driver_codes:
                            lap = session.laps.pick_driver(code).pick_fastest()

                            tel = lap.get_telemetry()
                            distance = tel['Distance']
                            speed = tel['Speed']

                            ax.plot(distance, speed, label=code)

                        ax.set_title("Fastest Lap Comparison")
                        ax.set_xlabel("Distance (m)")
                        ax.set_ylabel("Speed (km/h)")
                        ax.legend(title="Driver")
                        plt.tight_layout()

                        st.pyplot(fig)

                    except Exception as e:
                        st.write("")
    if format_selection == "Race":
        st.caption("Choosing individual laps coming soon")
        if st.button("generate"):
                if year and race and drivers_input:
                    try:
                        with st.spinner('please wait while your telemetry is loading'):
                            if drivers_input:
                                session = get_session(int(year), race, 'R')
                                session.load()
                                driver_names = [name.strip() for name in drivers_input.split(",")]
                                driver_codes = []
                                for name in driver_names:
                                    code = driver_name_to_code.get(name)
                                    if code:
                                        driver_codes.append(code)
                                    else:
                                        st.warning(f"Driver Not Recognized: {name}")
                                if len(driver_codes) < 1:
                                    st.warning("no valid drivers found")
                                else:
                                    fig, ax = plt.subplots()

                                    st.title("Sector Times")
                                    st.caption("Times taken from the fastest lap regardless of when throughout qualifying the drivers fastest lap was. Track development could play a factor if one driver made it further than another.")
                                    # Create an empty list to hold each driver's data
                        def format_time(timedelta_obj):
                            total_seconds = timedelta_obj.total_seconds()
                            minutes = int(total_seconds // 60)
                            seconds = total_seconds % 60
                            return f"{minutes}:{seconds:06.3f}"  # ensures leading zeros and 3 decimal places

                        rows = []

                        for code in driver_codes:
                            lap_data = session.laps.pick_driver(code).pick_fastest()

                            # Format each sector time
                            sector1 = format_time(lap_data['Sector1Time'])
                            sector2 = format_time(lap_data['Sector2Time'])
                            sector3 = format_time(lap_data['Sector3Time'])
                            lap_time = format_time(lap_data['LapTime'])

                            rows.append({
                                "Driver": code,
                                "Sector 1": sector1,
                                "Sector 2": sector2,
                                "Sector 3": sector3,
                                "Lap Time": lap_time
                            })

                        df = pd.DataFrame(rows)
                        st.dataframe(df)

                        fig, ax = plt.subplots()

                        st.title("Fastest Lap Comparison")
                        st.caption("Times taken from the fastest lap regardless of when throughout qualifying the drivers fastest lap was. Track development could play a factor if one driver made it further than another.")
                        for code in driver_codes:
                            lap = session.laps.pick_driver(code).pick_fastest()

                            tel = lap.get_telemetry()
                            distance = tel['Distance']
                            speed = tel['Speed']

                            ax.plot(distance, speed, label=code)

                        ax.set_title("Fastest Lap Comparison")
                        ax.set_xlabel("Distance (m)")
                        ax.set_ylabel("Speed (km/h)")
                        ax.legend(title="Driver")
                        plt.tight_layout()

                        st.pyplot(fig)

                    except Exception as e:
                        st.write("")

            


####COMPARE DRIVER####
############


############
####WHOLE RACES####
if option == "Whole Races":
    st.write("yeah i didn't get to this yet")