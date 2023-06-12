import streamlit as st
from multiapp import MultiApp
from apps import (
    airnode,
)

# st.set_page_config(layout="wide")


apps = MultiApp()

# Add all your application here
# apps.add_app("Timelapse", timelapse.app)
# apps.add_app("Home", home.app)
# apps.add_app("Create an interactive map", backends.app)
# apps.add_app("Customize the default map", customize.app)
# apps.add_app("Change basemaps", basemaps.app)
# apps.add_app("Add tile layers", tile_layer.app)
# apps.add_app("Add vector", vector.app)
# apps.add_app("Demo", demo.app)
# apps.add_app("Pydeck", uber_nyc.app)
apps.add_app("AirNode", airnode.app)
# The main app
apps.run()
