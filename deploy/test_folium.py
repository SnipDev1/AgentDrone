for m in ["jinja2", "branca", "folium"]:
    try:
        __import__(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: EXCEPTION {e}")
