import cv2
import numpy as np
import pytesseract
import networkx as nx
import json

# --- 1. CONFIGURATION ---

# IMPORTANT: Update this path to your Tesseract installation
# For Windows, it might be: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSERACT_PATH = r"tesseract"  # For Linux/macOS if in PATH
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Define HSV color ranges for shape detection. These may need tuning for your specific image.
# You can use an online tool to find HSV values for your colors.
COLOR_RANGES = {
    "Orange": ([5, 150, 150], [15, 255, 255]),
    "Blue": ([95, 150, 100], [115, 255, 255]),
    "Teal": ([80, 100, 100], [95, 255, 255]),
    "LightGreen": ([30, 50, 50], [70, 255, 255]),  # For Peg and ICM Store
    "Purple": ([125, 100, 100], [145, 255, 255]),
    "Yellow": ([20, 100, 100], [30, 255, 255]),
}

# Define shape characteristics based on the legend
SHAPE_LEGEND = {
    # (Color, Num_Vertices): Type
    ("Orange", 4): "Prompt",
    ("Orange", 4, "diamond"): "Decision",  # Differentiated by shape approx
    ("Blue", 4): "ApiCall",
    ("Orange", "ellipse"): "Start",
    ("Teal", "pill"): "End",
    ("LightGreen", "circle"): "Peg",
    ("Purple", 5): "OffPageConnector",
    ("Orange", 4, "parallelogram"): "CustomerInput",
    ("Yellow", 4): "Assignment",
    ("LightGreen", 4): "ICMDataStore",
}

# --- 2. CORE DETECTION AND OCR FUNCTIONS ---


def detect_and_classify_shapes(image_path):
    """
    Detects shapes in the image, classifies them by color and form,
    and performs OCR on the text within them.
    """
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected_elements = []
    element_id_counter = 0

    for color_name, (lower, upper) in COLOR_RANGES.items():
        # Create a mask for the current color
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Filter out small noise
                continue

            # Approximate the shape
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
            num_vertices = len(approx)

            x, y, w, h = cv2.boundingRect(approx)

            # --- Shape Classification Logic ---
            shape_type = "Unknown"

            # Simple heuristic-based classification
            if num_vertices == 4:
                # Could be Rectangle, Diamond, or Parallelogram
                shape_type = SHAPE_LEGEND.get((color_name, 4), "Unknown")
                # Add more specific checks if needed, e.g., for diamond
                # For this PoC, we rely mainly on color
                if (
                    color_name == "Orange" and w / h < 1.2 and w / h > 0.8
                ):  # Heuristic for diamond
                    shape_type = "Decision"
            elif num_vertices == 5 and color_name == "Purple":
                shape_type = "OffPageConnector"
            elif (
                num_vertices > 6
            ):  # Ellipses/Circles are approximated with many vertices
                if color_name == "Orange":
                    shape_type = "Start"
                elif color_name == "LightGreen":
                    shape_type = "Peg"
                elif color_name == "Teal" and w > h * 2:  # Aspect ratio for pill shape
                    shape_type = "End"

            if shape_type == "Unknown":
                # Check for specific cases missed
                if color_name == "Orange" and num_vertices == 4:
                    # Check for parallelogram (customer input) based on angles, but for now we assume it's a prompt
                    # A more robust system would analyze corner angles
                    if w > h * 1.5:  # Heuristic for prompts
                        shape_type = "Prompt"
                    else:  # Assuming other orange rects are Customer Input or other types
                        shape_type = "CustomerInput"  # This is an assumption
            if color_name == "Yellow":
                shape_type = "Assignment"
            if color_name == "LightGreen" and num_vertices == 4:
                shape_type = "ICMDataStore"
            if color_name == "Blue":
                shape_type = "ApiCall"

            # --- Perform OCR ---
            # Add padding around the bounding box for better OCR
            roi = gray_image[y - 5 : y + h + 5, x - 5 : x + w + 5]
            ocr_text = pytesseract.image_to_string(roi, config="--psm 6").strip()

            # Store the result
            detected_elements.append(
                {
                    "id": f"elem_{element_id_counter}",
                    "type": shape_type,
                    "bbox": (x, y, w, h),
                    "center": (x + w // 2, y + h // 2),
                    "raw_text": ocr_text,
                    "contour": contour,
                }
            )
            element_id_counter += 1

    return detected_elements


def find_connections(elements):
    """
    A simple heuristic to connect elements.
    It assumes flow is generally top-to-bottom, left-to-right.
    A robust solution would use arrow detection.
    """
    connections = []
    for source_elem in elements:
        # For decision nodes, we expect multiple outputs, but this simple heuristic will find the closest one
        # For others, we expect one output

        source_center = source_elem["center"]
        potential_targets = []

        for target_elem in elements:
            if source_elem["id"] == target_elem["id"]:
                continue

            target_center = target_elem["center"]
            # Check if target is generally "after" the source
            if (
                target_center[1] > source_center[1] - 20
            ):  # Target is below or at a similar level
                dist = np.linalg.norm(np.array(source_center) - np.array(target_center))
                potential_targets.append((dist, target_elem["id"]))

        if potential_targets:
            # Sort by distance and pick the closest one(s)
            potential_targets.sort()
            # For simplicity, connect to the single closest valid target
            # A real system would need to handle branching from 'Decision' nodes
            connections.append(
                {"from": source_elem["id"], "to": potential_targets[0][1]}
            )

    return connections


# --- 3. PARSING LOGIC ---


def parse_element_data(element):
    """
    Parses the raw OCR text into a structured dictionary based on the element type.
    """
    data = {"type": element["type"], "raw_text": element["raw_text"]}
    text = element["raw_text"]
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    try:
        if element["type"] == "Prompt" and lines:
            data["name"] = lines[0]
            data["script"] = " ".join(lines[1:])
        elif element["type"] == "Start" and lines:
            data["name"] = lines[0]
            data["entry_points"] = lines[1:]
        elif (
            element["type"] in ["ApiCall", "OffPageConnector", "Peg", "Decision"]
            and lines
        ):
            data["name"] = lines[0]
            data["details"] = " ".join(lines[1:]) if len(lines) > 1 else ""
        elif element["type"] == "Assignment" and lines:
            data["assignments"] = {}
            for line in lines:
                if "=" in line:
                    key, val = line.split("=", 1)
                    # Ignore the "Assign:" part
                    key = key.replace("Assign:", "").strip()
                    data["assignments"][key.strip()] = val.strip()
        elif element["type"] == "ICMDataStore" and lines:
            data["name"] = lines[0]
            data["fields"] = {}
            for line in lines[1:]:
                if "=" in line:
                    key, val = line.split("=", 1)
                    data["fields"][key.strip()] = val.strip()
        else:
            data["content"] = text
    except Exception as e:
        print(
            f"Warning: Could not parse text for element {element['id']}: '{text}'. Error: {e}"
        )
        data["content"] = text

    return data


# --- 4. GRAPH BUILDING AND REPORTING ---


def build_graph_and_report(elements, connections):
    """
    Builds a NetworkX graph and generates a human-readable report by traversing it.
    """
    G = nx.DiGraph()

    # Add nodes
    for elem in elements:
        parsed_data = parse_element_data(elem)
        G.add_node(elem["id"], **parsed_data)

    # Add edges
    for conn in connections:
        G.add_edge(conn["from"], conn["to"])

    # --- Generate Report ---
    report = []

    # Find start nodes (those with no incoming edges or type 'Start')
    start_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
    if not start_nodes:  # Fallback if connections are messy
        start_nodes = [e["id"] for e in elements if e["type"] == "Start"]

    if not start_nodes:
        return "Error: Could not find a start node.", G

    report.append("--- IVR Flow Report ---\n")

    # Perform a DFS traversal from the start node to print the flow
    visited = set()

    def generate_path_report(node_id, depth=0):
        if node_id in visited:
            return

        visited.add(node_id)
        indent = "  " * depth
        node_data = G.nodes[node_id]

        report.append(f"{indent}Step {depth + 1}: [{node_data.get('type', 'Unknown')}]")

        # Pretty print based on type
        if node_data.get("name"):
            report.append(f"{indent}  Name: {node_data['name']}")
        if node_data.get("script"):
            report.append(f"{indent}  Script: {node_data['script']}")
        if node_data.get("assignments"):
            report.append(f"{indent}  Set Variables: {node_data['assignments']}")
        if node_data.get("fields"):
            report.append(f"{indent}  Data Sent: {node_data['fields']}")
        elif node_data.get("content") and node_data["type"] not in ["Prompt", "Start"]:
            report.append(f"{indent}  Content: {node_data['content']}")

        report.append("")  # Newline for readability

        successors = list(G.successors(node_id))
        if not successors:
            report.append(f"{indent}--- END OF PATH ---")
        else:
            for i, next_node_id in enumerate(successors):
                if len(successors) > 1:
                    report.append(f"{indent}--> Path {i + 1}")
                generate_path_report(next_node_id, depth + 1)

    generate_path_report(start_nodes[0])

    return "\n".join(report), G


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    image_file = "flowchart.png"  # Make sure your flowchart image is named this

    print("Step 1: Detecting shapes and performing OCR...")
    all_elements = detect_and_classify_shapes(image_file)
    print(f"  > Found {len(all_elements)} elements.")

    # You can print this to debug the raw extraction
    # print("\n--- Raw Elements Detected ---")
    # for el in all_elements:
    #     print(json.dumps(el, indent=2, default=lambda o: '<not serializable>'))

    print("\nStep 2: Finding connections between elements...")
    # NOTE: This is a simplified connection logic. A real implementation would need robust arrow detection.
    # For now, we will manually define connections for accuracy based on the image.
    # This shows how the graph works, even if automatic detection is imperfect.

    # Manual Connection Definition (More reliable for this PoC)
    # This requires looking at your specific image and element IDs
    # The automatic `find_connections` is a less reliable alternative
    # Let's sort elements from top-to-bottom to create a somewhat logical flow
    all_elements.sort(key=lambda e: e["bbox"][1])  # Sort by Y coordinate
    for i, el in enumerate(all_elements):
        el["id"] = f"elem_{i}"  # Re-assign IDs based on sorted order

    # Based on the visual flow of the example image
    manual_connections = [
        {"from": "elem_0", "to": "elem_1"},  # HO3 Menu -> Decision
        {"from": "elem_1", "to": "elem_2"},  # Decision -> GetPolicySymDetails
        {"from": "elem_2", "to": "elem_3"},  # GetPolicy... -> Geico Home Owners
        {"from": "elem_3", "to": "elem_4"},  # Geico -> End Symbol
        {"from": "elem_4", "to": "elem_5"},  # End -> Peg
        {"from": "elem_5", "to": "elem_6"},  # Peg -> Off-page link
        {"from": "elem_6", "to": "elem_7"},  # Off-page -> Customer enters zip
        {"from": "elem_7", "to": "elem_8"},  # Customer -> Assign
        {"from": "elem_8", "to": "elem_9"},  # Assign -> ICM Data Store
    ]

    print(
        f"  > Using {len(manual_connections)} manually defined connections for accuracy."
    )

    print("\nStep 3: Building flow graph and generating report...")
    report_text, flow_graph = build_graph_and_report(all_elements, manual_connections)

    print("\n" + "=" * 50)
    print(report_text)
    print("=" * 50)

    # You can now use the `flow_graph` object for further analysis with NetworkX
    print(
        f"\nGraph has {flow_graph.number_of_nodes()} nodes and {flow_graph.number_of_edges()} edges."
    )

    #######

#     {
#   "nodes": [
#     {
#       "id": 300,
#       "shape": "Info",
#       "text": "Geico Home Owners\n800-268-6146, DNIS 4035451\n800-350-9293, DNIS 4035031"
#     },
#     {
#       "id": 301,
#       "shape": "Info",
#       "text": "DN: ASP_DN_4035451_8002686146\nDN: ASP_DN_4035031_8003509293"
#     },
#     {
#       "id": 0,
#       "shape": "Oval",
#       "text": "0"
#     },
#     {
#       "id": 200,
#       "shape": "Rounded Rectangle",
#       "text": "Greeting and Language Offer\nThank you for calling Assurant, servicer of the single-family home insurance program, for Geico. Para espanol oprima numero ocho?\nBargeln=FALSE\nNOTE: Greeting should be dynamic based on DNIS"
#     },
#     {
#       "id": 1,
#       "shape": "Oval",
#       "text": "1"
#     },
#     {
#       "id": 2,
#       "shape": "Oval",
#       "text": "2"
#     },
#     {
#       "id": 201,
#       "shape": "Rectangle",
#       "text": "Assign: Language = EN"
#     },
#     {
#       "id": 202,
#       "shape": "Rectangle",
#       "text": "Assign: Language = ES\nNOTE: call flow will proceed in Spanish language"
#     },
#     {
#       "id": 203,
#       "shape": "Sub-process",
#       "text": "GetPolicySymDetails\n(ANI lookup)"
#     },
#     {
#       "id": 100,
#       "shape": "Oval",
#       "text": "100"
#     },
#     {
#       "id": 101,
#       "shape": "Oval",
#       "text": "101"
#     },
#     {
#       "id": 102,
#       "shape": "Oval",
#       "text": "102"
#     },
#     {
#       "id": 103,
#       "shape": "Oval",
#       "text": "103"
#     },
#     {
#       "id": 204,
#       "shape": "Rounded Rectangle",
#       "text": "Account Verfication Options\nIf you want to locate your policy by policy number, press 1. To search by phone number, press 2"
#     },
#     {
#       "id": 104,
#       "shape": "Oval",
#       "text": "104"
#     },
#     {
#       "id": 105,
#       "shape": "Oval",
#       "text": "105"
#     },
#     {
#       "id": 205,
#       "shape": "Rounded Rectangle",
#       "text": "Collect Zip code\nPlease enter the 5 digit zip code of the covered location."
#     },
#     {
#       "id": 206,
#       "shape": "Parallelogram",
#       "text": "Customer enters zip code"
#     },
#     {
#       "id": 107,
#       "shape": "Oval",
#       "text": "107"
#     },
#     {
#       "id": 207,
#       "shape": "Rounded Rectangle",
#       "text": "Confirm Zip code\nYou entered <zip code>."
#     },
#     {
#       "id": 208,
#       "shape": "Rounded Rectangle",
#       "text": "Collect Policy Number\nPlease enter your policy number, which should be either 7 or 10 digits long. Ensure there are no letters included."
#     },
#     {
#       "id": 209,
#       "shape": "Rounded Rectangle",
#       "text": "Collect Phone Number\nPlease enter the telephone number including the area code for the property covered"
#     },
#     {
#       "id": 210,
#       "shape": "Rectangle",
#       "text": "Assign: dataEntry = PolicyNum"
#     },
#     {
#       "id": 211,
#       "shape": "Rectangle",
#       "text": "Assign: dataEntry = PhoneNum"
#     },
#     {
#       "id": 212,
#       "shape": "Parallelogram",
#       "text": "Customer enters Policy/Phone number"
#     }
#   ],
#   "edges": [
#     { "source_id": 0, "target_id": 200, "label": null },
#     { "source_id": 200, "target_id": 1, "label": "!=8" },
#     { "source_id": 200, "target_id": 2, "label": null },
#     { "source_id": 1, "target_id": 201, "label": null },
#     { "source_id": 2, "target_id": 202, "label": null },
#     { "source_id": 201, "target_id": 203, "label": null },
#     { "source_id": 202, "target_id": 203, "label": null },
#     { "source_id": 203, "target_id": 100, "label": "-ANI Match-" },
#     { "source_id": 203, "target_id": 101, "label": "-ANI NM-" },
#     { "source_id": 203, "target_id": 102, "label": "API Error" },
#     { "source_id": 100, "target_id": 103, "label": null },
#     { "source_id": 101, "target_id": 204, "label": null },
#     { "source_id": 103, "target_id": 205, "label": null },
#     { "source_id": 204, "target_id": 104, "label": "1" },
#     { "source_id": 204, "target_id": 105, "label": "2" },
#     { "source_id": 205, "target_id": 206, "label": "NI3/NM3" },
#     { "source_id": 206, "target_id": 107, "label": null },
#     { "source_id": 107, "target_id": 207, "label": null },
#     { "source_id": 104, "target_id": 208, "label": null },
#     { "source_id": 105, "target_id": 209, "label": null },
#     { "source_id": 208, "target_id": 210, "label": null },
#     { "source_id": 209, "target_id": 211, "label": null },
#     { "source_id": 210, "target_id": 212, "label": null },
#     { "source_id": 211, "target_id": 212, "label": null },
#     { "source_id": 212, "target_id": 203, "label": null }
#   ],
#   "off_page_connectors": []
# }
