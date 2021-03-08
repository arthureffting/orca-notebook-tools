import json
import os

year_mapping = {}

with open('result.json') as f:
    data = json.load(f)
    print(len(data["images"]), "images loaded")
    for image in data["images"]:
        image_data = {
            "lines": [],
            "path": image["path"],
            "year": image["path"].split("/")[0].split("_")[0],
            "filename": image["path"].split("/")[1]
        }
        if image_data["year"] not in year_mapping:
            year_mapping[image_data["year"]] = []
        for line in image["lines"]:
            image_data["lines"].append({
                "line": line["extracted_path"].split("/")[3].split(".")[0].split("-")[1],
                "text": line["predicted_text"],
                "start_position": [line["steps"][0]["x0"], line["steps"][0]["y0"]]
            })
        year_mapping[image_data["year"]].append(image_data)

for year in year_mapping:
    with open(os.path.join("", year + ".json"), 'w') as outfile:
        print("[dumping", len(year_mapping[year]), "images for year", year, "]")
        json.dump(year_mapping[year], outfile)
