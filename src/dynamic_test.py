# Simulated configuration boundary (e.g., maximum possible slots)
MAX_BOUND_POLES = 3  

user_input = input("\n[SETUP] Which specific pole IDs are present at this site? (e.g., 1,2,3 or 2,5): ")

# FALLBACK: If they press enter (empty string), default to all poles up to max boundary
if user_input.strip() == "":
    active_poles = list(range(1, MAX_BOUND_POLES + 1))
else:
    active_poles = [int(p.strip()) for p in user_input.split(",") if p.strip().isdigit()]

# Safe to run max() now because active_poles will never be empty
max_pole_id = max(active_poles) 

print(f"Locked in tracking for poles: {active_poles} (Max Array Index Slot: {max_pole_id})")

# Simulated clicks array (6 data clicks + 1 confirmation click = 7 points total)
# Let's say user clicked (x,y) coordinates sequentially from left to right
mock_points = [
    (10, 100), (10, 500),  # First pair clicked (idx 0)
    (30, 120), (30, 520),  # Second pair clicked (idx 1)
    (50, 110), (50, 510),  # Third pair clicked (idx 2)
    (999, 999)             # Confirmation click (idx 3)
]

# Process each designated pole ID horizontally
for idx, poleId in enumerate(active_poles):
    # Sequential layout pulls data clicks from the 0-indexed points array perfectly
    top = mock_points[2 * idx] 
    bottom = mock_points[2 * idx + 1]
    
    print(f"Loop Index (idx): {idx} | Target physical header: s{poleId}")
    print(f"  -> Pulling data clicks from mock_points[{2 * idx}] and mock_points[{2 * idx + 1}]")
    print(f"  -> Coordinates mapped: Top={top}, Bottom={bottom}\n")