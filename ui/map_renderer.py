import folium
from folium import plugins

def create_trip_folium_map(result_data, max_gps_markers=300):
    """
    Renders an interactive Folium map with:
    - Blue circle markers for raw GPS observations
    - Green PolyLine for Ground Truth reference route
    - Bright Red PolyLine for Reconstructed route
    - LayerControl, Legend, and Tile toggle
    """
    gps_coords = result_data.get("gps_coordinates", [])
    reconstructed_lines = result_data.get("reconstructed_route_lines", [])
    gt_lines = result_data.get("ground_truth_route_lines", [])
    
    if not gps_coords:
        m = folium.Map(location=[49.85, 8.58], zoom_start=13)
        return m
        
    # Center map at middle of GPS trajectory
    mid_idx = len(gps_coords) // 2
    center_lat, center_lon = gps_coords[mid_idx]
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron"
    )
    
    # Layer 1: Ground Truth Route (Green)
    if gt_lines:
        gt_group = folium.FeatureGroup(name="🟢 Ground Truth Route (.route)", show=True)
        for line in gt_lines:
            folium.PolyLine(
                locations=line,
                color="#10B981", # Emerald green
                weight=5,
                opacity=0.75,
                tooltip="Ground Truth Segment"
            ).add_to(gt_group)
        gt_group.add_to(m)
        
    # Layer 2: Reconstructed Route (Red / Deep Orange)
    if reconstructed_lines:
        rec_group = folium.FeatureGroup(name="🔴 PathSnap Reconstructed Route", show=True)
        for line in reconstructed_lines:
            folium.PolyLine(
                locations=line,
                color="#EF4444", # Vivid red
                weight=4,
                opacity=0.90,
                tooltip="PathSnap Reconstructed Segment"
            ).add_to(rec_group)
        rec_group.add_to(m)
        
    # Layer 3: Raw GPS Observations (Blue Dots)
    if gps_coords:
        gps_group = folium.FeatureGroup(name="🔵 Raw GPS Observations (.track)", show=True)
        # Downsample marker rendering if many points to keep browser snappy
        step = max(1, len(gps_coords) // max_gps_markers)
        for i in range(0, len(gps_coords), step):
            pt = gps_coords[i]
            folium.CircleMarker(
                location=pt,
                radius=3,
                color="#2563EB", # Blue
                fill=True,
                fill_color="#3B82F6",
                fill_opacity=0.7,
                tooltip=f"GPS Obs #{i}: [{pt[0]:.5f}, {pt[1]:.5f}]"
            ).add_to(gps_group)
        gps_group.add_to(m)
        
    # Add Start / End Markers
    if gps_coords:
        folium.Marker(
            location=gps_coords[0],
            popup="Trip Start",
            icon=folium.Icon(color="green", icon="play", prefix="fa")
        ).add_to(m)
        folium.Marker(
            location=gps_coords[-1],
            popup="Trip End",
            icon=folium.Icon(color="red", icon="stop", prefix="fa")
        ).add_to(m)
        
    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m
