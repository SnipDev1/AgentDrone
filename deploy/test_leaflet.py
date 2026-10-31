import streamlit as st
import streamlit.components.v1 as components

st.title("🛸 Leaflet Test")

map_html = """
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>#map { height: 400px; width: 100%; }</style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([54.90, 83.15], 9);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
        L.marker([54.90, 83.15]).addTo(map).bindPopup('<b>Novosibirsk</b>').openPopup();
    </script>
</body>
</html>
"""

components.html(map_html, height=420)
st.success("Components HTML works!")
